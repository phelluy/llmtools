"""
Utilitaire de conversion PDF vers Markdown utilisant l'API OCR de Mistral.

Ce script convertit des fichiers PDF en Markdown en utilisant l'API OCR de Mistral.
Il gère également le téléchargement automatique des images détectées dans les PDF.

Fonctionnalités:
- Conversion de PDF en Markdown via l'API Mistral OCR
- Téléchargement automatique des images depuis les serveurs Mistral
- Organisation des images dans des sous-dossiers dédiés (format: {nom_pdf}_images/)
- Mise à jour des liens markdown pour référencer les images locales

Usage:
    python pdf2md.py <dossier>

Exemple:
    python pdf2md.py ./mon_dossier_pdf

Prérequis:
- Variable d'environnement MISTRAL_API_KEY définie
- Fichiers PDF à convertir

Structure de sortie:
    mon_dossier/
    ├── document.pdf
    ├── document.md              # Markdown généré
    └── document_images/         # Images téléchargées
        ├── image_001.png
        ├── image_002.png
        └── ...
"""
import os
import sys
import json
import base64
import requests
import re
from urllib.parse import urlparse

API_KEY = os.environ.get("MISTRAL_API_KEY")  # Stocke ta clé API dans une variable d'environnement

def upload_file(pdf_path):
    """
    Upload un fichier PDF vers l'API Mistral pour traitement OCR.
    
    Args:
        pdf_path (str): Chemin absolu vers le fichier PDF à uploader
    
    Returns:
        str: L'ID du fichier uploadé sur les serveurs Mistral, ou None en cas d'échec
    """
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
    """
    Récupère une URL signée temporaire pour accéder au fichier uploadé.
    
    Args:
        file_id (str): L'identifiant du fichier retourné par upload_file
    
    Returns:
        str: URL signée valide pendant 24h permettant d'accéder au fichier, ou None en cas d'échec
    """
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
    """
    Lance le traitement OCR sur un document PDF via l'API Mistral.
    
    Args:
        document_url (str): URL signée du document à traiter
    
    Returns:
        dict: Résultat JSON de l'OCR contenant les pages et le markdown extrait, ou None en cas d'échec
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral-ocr-latest",
        "include_image_base64": True,
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

def save_image_from_base64(image_base64, output_path):
    """
    Décode et sauvegarde une image depuis son encodage base64.
    
    Args:
        image_base64 (str): Image encodée en base64 (format: "data:image/jpeg;base64,...")
        output_path (str): Chemin local où sauvegarder l'image
    
    Returns:
        bool: True si la sauvegarde a réussi, False sinon
    """
    try:
        # Extraire les données base64 (enlever le préfixe "data:image/...;base64,")
        if ',' in image_base64:
            base64_data = image_base64.split(',', 1)[1]
        else:
            base64_data = image_base64
        
        # Décoder et sauvegarder
        image_bytes = base64.b64decode(base64_data)
        with open(output_path, 'wb') as f:
            f.write(image_bytes)
        return True
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")
        return False

def extract_images_from_ocr(ocr_json):
    """
    Extrait toutes les images du résultat OCR avec leurs données base64.
    
    Args:
        ocr_json (dict): Résultat JSON retourné par l'API OCR Mistral
    
    Returns:
        list: Liste de dictionnaires contenant {"id": str, "base64": str} pour chaque image
    """
    images = []
    if "pages" not in ocr_json:
        return images
    
    for page in ocr_json["pages"]:
        if "images" in page and isinstance(page["images"], list):
            for img in page["images"]:
                if "id" in img and "image_base64" in img:
                    images.append({
                        "id": img["id"],
                        "base64": img["image_base64"]
                    })
    return images

def process_images(markdown, images_list, images_dir, pdf_name):
    """
    Sauvegarde les images depuis le base64 et met à jour les références dans le markdown.
    
    Args:
        markdown (str): Le texte markdown contenant des références d'images
        images_list (list): Liste des images extraites du JSON OCR avec id et base64
        images_dir (str): Le répertoire où sauvegarder les images
        pdf_name (str): Le nom du PDF (sans extension) pour nommer le sous-dossier
    
    Returns:
        str: Le markdown avec les liens d'images mis à jour vers les fichiers locaux
    """
    if not images_list:
        return markdown
    
    # Créer le sous-dossier pour les images de ce PDF
    pdf_images_dir = os.path.join(images_dir, f"{pdf_name}_images")
    os.makedirs(pdf_images_dir, exist_ok=True)
    
    updated_markdown = markdown
    image_counter = 1
    
    for img in images_list:
        img_id = img["id"]
        img_base64 = img["base64"]
        
        # Déterminer l'extension depuis l'ID (ex: "img-6.jpeg")
        if '.' in img_id:
            extension = '.' + img_id.split('.')[-1]
        else:
            extension = '.png'
        
        # Nom de fichier local pour l'image
        local_image_name = f"image_{image_counter:03d}{extension}"
        local_image_path = os.path.join(pdf_images_dir, local_image_name)
        
        # Sauvegarder l'image depuis base64
        print(f"  Saving image {image_counter}: {img_id}")
        if save_image_from_base64(img_base64, local_image_path):
            # Mettre à jour les références dans le markdown
            relative_path = os.path.join(f"{pdf_name}_images", local_image_name)
            # Remplacer toutes les occurrences de l'ID de l'image par le chemin local
            updated_markdown = updated_markdown.replace(img_id, relative_path)
            print(f"  Saved as {relative_path}")
            image_counter += 1
        else:
            print(f"  Failed to save image {img_id}")
    
    return updated_markdown

def extract_markdown(ocr_json):
    """
    Extrait et concatène le contenu markdown de toutes les pages du résultat OCR.
    
    Args:
        ocr_json (dict): Résultat JSON retourné par l'API OCR Mistral
    
    Returns:
        str: Texte markdown concatené de toutes les pages, séparées par des sauts de ligne
    """
    if "pages" not in ocr_json or not ocr_json["pages"]:
        return ""
    return "\n".join(page.get("markdown", "") for page in ocr_json["pages"])

def process_pdf(pdf_path, md_path):
    """
    Traite un fichier PDF complet: upload, OCR, extraction markdown et téléchargement des images.
    
    Cette fonction orchestre toutes les étapes de conversion:
    1. Upload du PDF vers Mistral
    2. Récupération d'une URL signée
    3. Exécution de l'OCR
    4. Extraction du markdown
    5. Téléchargement et organisation des images
    6. Sauvegarde du fichier markdown final
    
    Args:
        pdf_path (str): Chemin absolu vers le fichier PDF source
        md_path (str): Chemin absolu où sauvegarder le fichier markdown résultant
    """
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
    
    # Extraire et sauvegarder les images
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    images_dir = os.path.dirname(md_path)
    images_list = extract_images_from_ocr(ocr_result)
    markdown = process_images(markdown, images_list, images_dir, pdf_name)
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"Wrote {md_path}")

def main(root_dir):
    """
    Point d'entrée principal du script. Parcourt récursivement un dossier et convertit tous les PDF.
    
    Cette fonction:
    - Vérifie la présence de la clé API
    - Parcourt récursivement le dossier racine
    - Traite chaque fichier PDF trouvé
    - Ignore les PDF déjà convertis (si le .md existe)
    
    Args:
        root_dir (str): Chemin du dossier racine à parcourir
    """
    if not API_KEY:
        print("❌ API key manquante. Définis MISTRAL_API_KEY dans les variables d'environnement.")
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
