#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sys
from pathlib import Path

# Must be set before importing docling/huggingface_hub.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def build_converter(
    model_id: str,
    timeout_seconds: int | None = None,
    force_transformers: bool = False,
):
    """Build a Docling PDF converter configured for a VLM model.

    This function targets current Docling APIs and raises a clear error when
    an incompatible Docling version is installed.
    """
    try:
        from docling.document_converter import (
            DocumentConverter,
            InputFormat,
            PdfFormatOption,
        )
        from docling.datamodel.pipeline_options import PdfPipelineOptions, VlmPipelineOptions
        from docling.datamodel.vlm_engine_options import (
            TransformersVlmEngineOptions,
            VlmEngineType,
        )
        from docling.pipeline.vlm_pipeline import VlmPipeline
    except Exception as exc:
        raise RuntimeError(
            "Could not import Docling classes required by this script. "
            "Please ensure docling is installed in this venv.\n"
            f"Import error: {exc}"
        ) from exc

    vlm_pipeline_options = VlmPipelineOptions()

    if timeout_seconds is not None and hasattr(vlm_pipeline_options, "document_timeout"):
        vlm_pipeline_options.document_timeout = timeout_seconds

    if force_transformers and hasattr(vlm_pipeline_options, "vlm_options"):
        engine_options = getattr(vlm_pipeline_options.vlm_options, "engine_options", None)
        if engine_options is not None and hasattr(engine_options, "engine_type"):
            vlm_pipeline_options.vlm_options.engine_options = TransformersVlmEngineOptions(
                engine_type=VlmEngineType.TRANSFORMERS
            )

    # Newer Docling: model is configured under vlm_options.model_spec.default_repo_id.
    if (
        hasattr(vlm_pipeline_options, "vlm_options")
        and hasattr(vlm_pipeline_options.vlm_options, "model_spec")
        and hasattr(vlm_pipeline_options.vlm_options.model_spec, "default_repo_id")
    ):
        vlm_pipeline_options.vlm_options.model_spec.default_repo_id = model_id
    # Older Docling fallbacks.
    elif hasattr(vlm_pipeline_options, "model"):
        setattr(vlm_pipeline_options, "model", model_id)
    elif hasattr(vlm_pipeline_options, "model_id"):
        setattr(vlm_pipeline_options, "model_id", model_id)
    else:
        raise RuntimeError(
            "Your installed Docling version does not expose a known VLM model field. "
            "Expected one of: `vlm_options.model_spec.default_repo_id`, `model`, or `model_id`."
        )

    # Prefer explicit VLM pipeline when available in newer Docling versions.
    format_options = {
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=vlm_pipeline_options,
        )
    }
    return DocumentConverter(format_options=format_options)


def extract_and_save_images(document_dict: dict, output_dir: Path) -> dict[str, str]:
    """Extract images from Docling document JSON and save them to disk.
    
    Returns a mapping of old image URI to new local paths.
    """
    images_dir = output_dir / "images"
    image_mapping = {}
    
    # Extract images from the document dict structure
    if "pictures" not in document_dict:
        return image_mapping
    
    image_count = 0
    
    for picture in document_dict["pictures"]:
        try:
            if "image" not in picture or "uri" not in picture["image"]:
                continue
            
            uri = picture["image"]["uri"]
            mime_type = picture["image"].get("mimetype", "image/png")
            
            # Determine file extension from mime type
            ext_map = {
                "image/png": "png",
                "image/jpeg": "jpg",
                "image/jpg": "jpg",
                "image/webp": "webp",
            }
            extension = ext_map.get(mime_type, "png")
            
            # Check if it's a base64 data URI
            if uri.startswith("data:"):
                # Extract base64 data
                try:
                    # Format: data:image/png;base64,<base64_string>
                    base64_data = uri.split(",", 1)[1]
                    image_data = base64.b64decode(base64_data)
                except Exception as e:
                    print(f"[warning] Failed to decode base64 image: {e}")
                    continue
                
                image_count += 1
                images_dir.mkdir(parents=True, exist_ok=True)
                
                # Save image
                image_filename = f"image_{image_count:03d}.{extension}"
                image_path = images_dir / image_filename
                image_path.write_bytes(image_data)
                
                # Track the image for reference updates
                local_path = f"images/{image_filename}"
                image_mapping[uri] = local_path
                print(f"[info] Extracted image: {image_path}")
        except Exception as e:
            print(f"[warning] Failed to extract image: {e}")
            continue
    
    return image_mapping


def convert_pdf(
    input_pdf: Path,
    output_dir: Path,
    model_id: str,
    force_transformers: bool = False,
) -> None:
    if not input_pdf.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

    output_dir.mkdir(parents=True, exist_ok=True)

    timeout_seconds = int(os.environ.get("DOCLING_TIMEOUT_SECONDS", "0")) or None

    print(f"[info] HF_HUB_DISABLE_XET={os.environ.get('HF_HUB_DISABLE_XET')}")
    print(f"[info] HF token detected={'yes' if os.environ.get('HF_TOKEN') else 'no'}")
    print(f"[info] force_transformers={'yes' if force_transformers else 'no'}")
    print(f"[info] Building converter (model={model_id})")
    converter = build_converter(
        model_id=model_id,
        timeout_seconds=timeout_seconds,
        force_transformers=force_transformers,
    )
    print("[info] Starting conversion...")
    result = converter.convert(str(input_pdf))

    stem = input_pdf.stem

    # Markdown output
    md_path = output_dir / f"{stem}.md"
    md_text = result.document.export_to_markdown()
    md_path.write_text(md_text, encoding="utf-8")

    # Structured JSON output
    json_path = output_dir / f"{stem}.json"
    if hasattr(result.document, "export_to_dict"):
        payload = result.document.export_to_dict()
    elif hasattr(result.document, "to_dict"):
        payload = result.document.to_dict()
    else:
        # Safe fallback with minimal metadata
        payload = {
            "source": str(input_pdf),
            "note": "No export_to_dict/to_dict method found on document object.",
            "markdown_path": str(md_path),
        }

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Extract and save images
    print("[info] Extracting images...")
    image_mapping = extract_and_save_images(payload, output_dir)
    
    # Update markdown with local image references
    if image_mapping:
        for old_uri, new_path in image_mapping.items():
            # Replace any reference to the data URI with local path
            md_text = md_text.replace(f"(data:", f"({new_path}")
            # More lenient replacement in case URI appears differently
            md_text = re.sub(r"\!\[.*?\]\s*\(\s*data:[^)]*\)", f"![Image]({new_path})", md_text)
        
        md_path.write_text(md_text, encoding="utf-8")

    print(f"Done. Markdown: {md_path}")
    print(f"Done. JSON:     {json_path}")
    if image_mapping:
        print(f"Done. Images:   {output_dir / 'images'} ({len(image_mapping)} image(s))")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a PDF with Docling using a Granite VLX model."
    )
    parser.add_argument(
        "input_pdf",
        type=Path,
        help="Path to the input PDF file.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("./docling_output"),
        help="Directory where output files are written (default: ./docling_output).",
    )
    parser.add_argument(
        "--model-id",
        default="ibm-granite/granite-docling-258M",
        help=(
            "Hugging Face model id for Granite VLX/VLM in Docling. "
            "Override this if your setup uses a different Granite model id."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=900,
        help=(
            "Hard timeout for one document in seconds (default: 900). "
            "Use 0 to disable timeout."
        ),
    )
    parser.add_argument(
        "--verbose-download",
        action="store_true",
        help="Enable verbose logs for Hugging Face/Docling download and model loading steps.",
    )
    parser.add_argument(
        "--force-transformers",
        action="store_true",
        help="Force transformers inference engine instead of auto-select (MLX on Apple Silicon).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["DOCLING_TIMEOUT_SECONDS"] = str(args.timeout_seconds)

    if args.verbose_download:
        os.environ["HF_HUB_VERBOSITY"] = "debug"
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
        logging.getLogger("huggingface_hub").setLevel(logging.DEBUG)
        logging.getLogger("docling").setLevel(logging.INFO)
        print("[info] Verbose download logging enabled")

    try:
        convert_pdf(
            input_pdf=args.input_pdf,
            output_dir=args.output_dir,
            model_id=args.model_id,
            force_transformers=args.force_transformers,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
