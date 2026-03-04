import os
import sys
import json
import requests

API_KEY = os.environ.get("MISTRAL_API_KEY")  # Stocke ta clé API dans une variable d'environnement

def upload_file(pdf_path):
    with open(pdf_path, 'rb') as f:
        files = {
            'purpose': (None, 'ocr'),
            'file': (os.path.basename(pdf_path), f, 'application/pdf')
        }
        headers = {"Authorization": f"Bearer {API_KEY}"}
        response = requests.post("https://api.mistral.ai/v1/files", headers=headers, files=files)
    if response.status_code != 200:
        print(f"Upload failed for {pdf_path}: {response.text}")
        return None
    return response.json().get("id")

def get_signed_url(file_id):
    url = f"https://api.mistral.ai/v1/files/{file_id}/url?expiry=24"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to get signed URL: {response.text}")
        return None
    return response.json().get("url")

def run_ocr(document_url):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral-ocr-latest",
        "include_image_base64": False,
        "document": {
            "type": "document_url",
            "document_url": document_url
        }
    }
    response = requests.post("https://api.mistral.ai/v1/ocr", headers=headers, data=json.dumps(payload))
    if response.status_code != 200:
        print(f"OCR failed: {response.text}")
        return None
    return response.json()

def extract_markdown(ocr_json):
    if "pages" not in ocr_json or not ocr_json["pages"]:
        return ""
    return "\n".join(page.get("markdown", "") for page in ocr_json["pages"])

def process_pdf(pdf_path, md_path):
    print(f"Processing {pdf_path}")
    file_id = upload_file(pdf_path)
    if not file_id:
        return

    doc_url = get_signed_url(file_id)
    if not doc_url:
        return

    ocr_result = run_ocr(doc_url)
    if not ocr_result:
        return

    markdown = extract_markdown(ocr_result)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"Wrote {md_path}")

def main(root_dir):
    if not API_KEY:
        print("❌ API key manquante. Définis MISTRAL_API_KEY dans les variables d’environnement.")
        sys.exit(1)

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(dirpath, filename)
                md_path = os.path.splitext(pdf_path)[0] + ".md"
                if os.path.exists(md_path):
                    print(f"Skip (exists): {md_path}")
                    continue
                process_pdf(pdf_path, md_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pdf2md.py <dossier>")
        sys.exit(1)
    main(sys.argv[1])
