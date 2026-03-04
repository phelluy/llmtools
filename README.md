# PDF to Markdown Converters

Two tools to convert PDF files to Markdown with image extraction. An online tool using the Mistral OCR API, and a local tool using IBM's Granite VLM model via Docling.

## Installation

### 1. Create a virtual environment with uv

```bash
uv venv
source .venv/bin/activate
```

Or directly without activating:
```bash
uv venv && source .venv/bin/activate
```

### 2. Install dependencies

```bash
uv pip install -r requirements.txt
```

**Dependencies:**
- `requests` - For API calls
- `docling` - PDF converter with VLM (Granite)
- `pillow` - Image processing
- `huggingface-hub` - Download models
- `mlx` - Inference on Apple Silicon (recommended for Mac)
- `mlx-vlm` - VLM models for MLX

## Available Tools

### 1. Docling PDF to Markdown (`docling-pdf2md.py`)

Converts a PDF to Markdown using IBM's Granite VLX model via Docling. **Recommended for documents that must remain local and private.**

#### Usage

```bash
python docling-pdf2md.py <pdf_path> -o <output_dir>
```

#### Examples

```bash
# Basic usage
python docling-pdf2md.py document.pdf -o output

# With custom timeout (in seconds)
python docling-pdf2md.py document.pdf -o output --timeout-seconds 300

# Force Transformers engine instead of MLX
python docling-pdf2md.py document.pdf -o output --force-transformers

# Verbose logs for model download
python docling-pdf2md.py document.pdf -o output --verbose-download
```

#### Output

The output directory contains:
- `<name>.md` - Converted Markdown with local image references
- `<name>.json` - Complete Docling structure in JSON
- `images/` - Images extracted from the PDF

#### Example: `test/docling_output/`

```
docling_output/
├── sample.md          # Markdown with image references
├── sample.json        # Structured data
└── images/
    └── image_001.png  # Extracted image
```

---

### 2. Mistral OCR to Markdown (`mistral-pdf2md.py`)

Converts a PDF to Markdown using the Mistral OCR API. **Fast and efficient but requires an API key and data are sent to Mistral's servers.**

#### Configuration

Before running the script, set the Mistral API key:

```bash
export MISTRAL_API_KEY="your-api-key"
```

#### Usage

```bash
python mistral-pdf2md.py <directory>
```

Recursively scans the directory for all `.pdf` files and creates a corresponding `.md` file.

#### Example

```bash
# Convert all PDFs in the 'documents' folder
python mistral-pdf2md.py documents

# Convert PDFs in the current directory
python mistral-pdf2md.py .
```

#### Output

For each PDF, generates:
- `<name>.md` - Extracted Markdown
- `sample_images/` - Extracted images (if present)

---

## Quick Test

A demo is available in the `test/` folder:

```bash
# Test PDF is already generated
ls test/sample.pdf

# Test Docling
python docling-pdf2md.py test/sample.pdf -o test/docling_output

# Test Mistral (requires MISTRAL_API_KEY)
python mistral-pdf2md.py test
```

---

## Troubleshooting

### ⚠️ Warning: `mx.metal.device_info is deprecated`

This is an internal MLX warning, not an error. The script works correctly.

### ⚠️ Error: `Could not import Docling classes`

Docling is not installed. Reinstall it:
```bash
uv pip install docling --force-reinstall
```

### ⚠️ Error: `No API key found for Mistral`

Set the API key before running:
```bash
export MISTRAL_API_KEY="your-key"
```

---

## Tools Comparison

| Feature | Docling | Mistral |
|---------|---------|---------|
| Image extraction | ✅ Yes | ✅ Yes |
| OCR | ✅ Yes (VLM) | ✅ Yes |
| Scanned PDFs | ✅ Good | ✅ Excellent |
| Cost | ✅ Free | ⚠️ API paid |
| Installation | ⚠️ Heavy | ✅ Light |
| Speed | ✅ Fast | ⚠️ Network calls |
| Apple Silicon | ✅ MLX native | ℹ️ Network |

---

## Documentation

- [Docling Documentation](https://github.com/DS4SD/docling)
- [Mistral API Documentation](https://docs.mistral.ai/)
