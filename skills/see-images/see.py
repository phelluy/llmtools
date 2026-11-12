#!/usr/bin/env python3
"""Describe one or more images using the Unistra qwen3 multimodal model.

Calls the OpenAI-compatible chat completions endpoint via curl.

Environment:
  UNISTRA_API_KEY       API key (required) — matches provider "unistra" in config.toml
  SEE_IMAGES_MODEL      Model id (default: qwen3)
  SEE_IMAGES_API_BASE   Base URL (default: https://conversation.ia.unistra.fr/api)

Examples:
  see.py photo.png
  see.py photo.png -p "Que contient ce schema ?"
  see.py a.png b.png --model qwen3
  see.py shot.png --crop 200,20,700,140 --scale 3 -p "Transcris le titre"
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import subprocess
import sys
import tempfile

DEFAULT_API_BASE = "https://conversation.ia.unistra.fr/api"
DEFAULT_MODEL = "qwen3"
DEFAULT_PROMPT = "Décris cette image en détail."
API_KEY_ENV = "UNISTRA_API_KEY"


def image_mime_type(path: str) -> str:
    """Best-effort MIME type for an image path, keyed on extension."""
    mime, _ = mimetypes.guess_type(path)
    if mime and mime.startswith("image/"):
        return mime
    ext = os.path.splitext(path)[1].lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }.get(ext, "image/png")


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def parse_crop(value: str) -> tuple[int, int, int, int]:
    """Parse "x0,y0,x1,y1" from the command line."""
    try:
        x0, y0, x1, y1 = (int(part) for part in value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("attendu : x0,y0,x1,y1 en pixels") from None
    if x1 <= x0 or y1 <= y0:
        raise argparse.ArgumentTypeError("x1 doit etre > x0 et y1 > y0")
    return x0, y0, x1, y1


def prepare_image(
    path: str,
    crop: tuple[int, int, int, int] | None,
    scale: int,
) -> str:
    """Return the path of a cropped/enlarged PNG copy, or the path itself.

    Sending a crop is the cheapest quality win on a screenshot: the model
    loses 12-14 px text on a full-window capture and reads it fine once the
    region is enlarged.
    """
    if crop is None and scale == 1:
        return path
    try:
        from PIL import Image
    except ImportError:
        print(
            "Pillow est requis pour --crop/--scale : "
            "uv run --with pillow python3 see.py ...",
            file=sys.stderr,
        )
        sys.exit(1)
    image = Image.open(path)
    if crop is not None:
        image = image.crop(crop)
    if scale != 1:
        image = image.resize(
            (image.width * scale, image.height * scale), Image.LANCZOS
        )
    out = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    out.close()
    image.convert("RGB").save(out.name)
    return out.name


def build_payload(model: str, prompt: str, images: list[str]) -> dict:
    content: list[dict] = [{"type": "text", "text": prompt}]
    for path in images:
        data_url = "data:{mime};base64,{b64}".format(
            mime=image_mime_type(path), b64=encode_image(path)
        )
        content.append({"type": "image_url", "image_url": {"url": data_url}})
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
    }


def call_api(api_base: str, api_key: str, payload: dict) -> str:
    url = api_base.rstrip("/") + "/chat/completions"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(payload, tmp)
        tmp_path = tmp.name
    try:
        cmd = [
            "curl", "-sS", "--max-time", "180",
            "-X", "POST", url,
            "-H", f"Authorization: Bearer {api_key}",
            "-H", "Content-Type: application/json",
            "--data-binary", f"@{tmp_path}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        os.unlink(tmp_path)

    if result.returncode != 0:
        print(f"curl failed (exit {result.returncode}): {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return result.stdout


def parse_response(raw: str) -> str:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(f"non-JSON response:\n{raw}", file=sys.stderr)
        sys.exit(1)

    if "error" in data:
        print(f"API error:\n{json.dumps(data['error'], ensure_ascii=False, indent=2)}", file=sys.stderr)
        sys.exit(1)

    choices = data.get("choices") or []
    if not choices:
        print(f"no choices in response:\n{raw}", file=sys.stderr)
        sys.exit(1)

    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Describe images with the Unistra qwen3 multimodal model."
    )
    parser.add_argument("images", nargs="+", help="one or more image file paths")
    parser.add_argument(
        "-p", "--prompt",
        default=DEFAULT_PROMPT,
        help="prompt sent with the image (default: detailed French description)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("SEE_IMAGES_MODEL", DEFAULT_MODEL),
        help="model id (default: qwen3)",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("SEE_IMAGES_API_BASE", DEFAULT_API_BASE),
        help="API base URL (default: https://conversation.ia.unistra.fr/api)",
    )
    parser.add_argument(
        "--crop",
        type=parse_crop,
        default=None,
        metavar="X0,Y0,X1,Y1",
        help="recadrer avant l'envoi (pixels de l'image source)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        metavar="N",
        help="agrandir l'image d'un facteur N (2-4 pour du petit texte)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=f"API key (default: ${API_KEY_ENV})",
    )
    args = parser.parse_args(argv)

    api_key = args.api_key or os.environ.get(API_KEY_ENV, "")
    if not api_key:
        print(
            f"{API_KEY_ENV} is not set. Export it before running, or pass --api-key.",
            file=sys.stderr,
        )
        return 1

    for path in args.images:
        if not os.path.isfile(path):
            print(f"image not found: {path}", file=sys.stderr)
            return 1

    variants = [prepare_image(path, args.crop, args.scale) for path in args.images]
    try:
        payload = build_payload(args.model, args.prompt, variants)
        raw = call_api(args.api_base, api_key, payload)
    finally:
        for original, variant in zip(args.images, variants):
            if variant != original:
                os.unlink(variant)
    print(parse_response(raw))
    return 0


if __name__ == "__main__":
    sys.exit(main())
