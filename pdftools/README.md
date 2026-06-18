# PDF to Markdown Converters

Three tools to convert PDF files to Markdown with image extraction: Mistral OCR API, UNISTRA Qwen vision API (incremental page-by-page), and a local tool using IBM's Granite VLM model via Docling.

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
- `docling` - PDF converter with VLM (Granite, optional, for local conversion)
- `pillow` - Image processing
- `huggingface-hub` - Download models (optional, for docling)
- `mlx` - Inference on Apple Silicon (optional, for docling)
- `mlx-vlm` - VLM models for MLX (optional, for docling)

## Available Tools

### 1. Mistral OCR to Markdown (`mistral-pdf2md.py`)

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

### 2. UNISTRA Qwen to Markdown (`unistra-pdf2md.py`)

Converts PDFs to Markdown using the UNISTRA Qwen vision model via the `/v1/chat/completions` API. **Uses an incremental approach: each page is sent to the LLM with the accumulated context of previous pages for better OCR quality.**

#### Configuration

Before running the script, set the UNISTRA API key:

```bash
export UNISTRA_API_KEY="your-api-key"
```

#### Usage

```bash
python unistra-pdf2md.py <pdf_path>              # Single file
python unistra-pdf2md.py <directory>              # Batch mode (recursive)
python unistra-pdf2md.py <pdf_path> --timeout 300 # Custom timeout per page
```

#### Algorithm

1. Each PDF page is converted to a JPEG image (base64).
2. The first page is sent to the LLM with the system prompt.
3. For each subsequent page, the image is sent along with the accumulated markdown of all previous pages as context.
4. The LLM extracts text between `<markdown>` and `</markdown>` tags.
5. Results are accumulated and written to the final `.md` file.

This incremental context improves transcription quality by giving the LLM the full document context as it processes each page.

#### Batch mode

When given a directory, the script recursively finds all `.pdf` files. PDFs that already have a corresponding `.md` file are skipped.

#### Output

For each PDF, generates:
- `<name>.md` - Extracted Markdown (same directory as the PDF)
- `<name>_images/` - Empty directory reserved for future image extraction

#### Example

```bash
# Convert all PDFs in a folder
python unistra-pdf2md.py ./documents/

# First run: converts all PDFs
# Second run: skips already converted PDFs

# Single file
python unistra-pdf2md.pdf ./important.pdf
```

---

## Quick Test

End-to-end tests are in the `tests/` folder:

```bash
# Run all tests (requires UNISTRA_API_KEY for full test)
uv run python tests/test_unistra.py
```

Tests cover:
- Single PDF conversion (1 page)
- Multi-page PDF (3 pages)
- Batch mode with skip detection

```bash
# Test PDFs are generated in tests/test_output/
ls tests/test_output/
```

---

## Troubleshooting

### ⚠️ Warning: `mx.metal.device_info is deprecated`

This is an internal MLX warning, not an error. The script works correctly.

### ⚠️ Error: `No API key found for Mistral`

Set the API key before running:
```bash
export MISTRAL_API_KEY="your-key"
```

### ⚠️ Error: `No API key found for UNISTRA`

Set the API key before running:
```bash
export UNISTRA_API_KEY="your-key"
```

---

## Tools Comparison

| Feature | Mistral | UNISTRA |
|---------|---------|---------|
| Image extraction | ✅ Yes | ❌ No (future) |
| OCR | ✅ Yes | ✅ Yes |
| Scanned PDFs | ✅ Excellent | ✅ Excellent |
| Cost | ⚠️ API paid | ⚠️ API paid |
| Installation | ✅ Light | ✅ Light |
| Speed | ⚠️ Network calls | ⚠️ Network calls |
| Context accumulation | ❌ No | ✅ Yes (page-by-page) |

---

## Documentation

- [Mistral API Documentation](https://docs.mistral.ai/)
- [UNISTRA Qwen API](https://github.com/unistra/qwen-inference-api)
- [Docling Documentation](https://github.com/DS4SD/docling) (optional, local conversion)
