# Flask Unified Server

Flask application for product classification, AI attributes extraction, and translation.

## Files

### 1. `app.py`
Main Flask server (port 6002). Combines:
- Category Classification
- AI Attributes Extraction
- English → Arabic Translation

### 2. `mapping.py`
Category hierarchies data for classification

### 3. `ai_att_mapping.py`
Attribute templates for different product types

### 4. `requirements.txt`
Python package dependencies

### 5. `.env.example`
Template for environment variables (copy to `.env`)

## Setup

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate (Windows)
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file
cp .env.example .env
# Then edit .env and add your API keys

# 5. Run server
python app.py
```

Server starts on: `http://localhost:6002`

## Test

```bash
curl http://localhost:6002/health
```

## API Endpoints

### Category Classification

- **POST** `/api/category/classify` - Single item classification
- **POST** `/api/category/classify-csv` - Process CSV file
- **POST** `/api/category/classify-batch` - Parallel batch classification (multiprocessing)

### AI Attributes Extraction

- **POST** `/api/attributes/extract` - Extract AI attributes for single item
- **POST** `/api/attributes/extract-batch` - Parallel batch AI attribute extraction (multiprocessing)

### Grammar Check

- **POST** `/api/grammar/check` - Grammar check for text (supports single and batch processing)
  - Uses LanguageTool locally (no external service required)
  - Returns corrected text with original text and changes summary
  - Supports both single text and batch array processing
  - Works independently without requiring GradProject

### Translation

- **POST** `/api/translation/translate` - English → Arabic translation

### Pipeline (Combined Workflow)

- **POST** `/api/pipeline/process` - End-to-end: Classification → Attributes → Translation
- **POST** `/api/pipeline/process-batch` - Batch pipeline processing

### Global

- **GET** `/health` - Global health check

For detailed examples, see [ENDPOINT_SAMPLES.md](ENDPOINT_SAMPLES.md)

## How It Works

```
app.py
  ├── Imports mapping.py          → Category hierarchies
  ├── Imports ai_att_mapping.py   → Attribute templates
  ├── Uses .env                    → API keys
  ├── Uses LanguageTool           → Local grammar checking
  └── Optionally uses GradProject → Image-based AI predictions (color/material)

User Request → app.py → Uses mappings + API keys → Returns result
Grammar Check → app.py → LanguageTool (local) → Returns corrected text
AI Attributes → app.py → OpenAI + Optional GradProject → Returns attributes
```

## File Dependencies

| File | Depends On |
|------|------------|
| `app.py` | `mapping.py`, `ai_att_mapping.py`, `.env`, `requirements.txt` |
| `mapping.py` | None (standalone data) |
| `ai_att_mapping.py` | None (standalone data) |
| `requirements.txt` | None (package list) |
| `.env` | None (configuration) |

## Multiprocessing Support

The server supports parallel processing for better performance:

- **Category Classification Batch**: Process multiple items concurrently using ThreadPoolExecutor
  - Endpoint: `/api/category/classify-batch`
  - Control workers with `max_workers` parameter (default: 3)

- **AI Attributes Batch**: Extract attributes for multiple items in parallel
  - Endpoint: `/api/attributes/extract-batch`
  - Control workers with `max_workers` parameter (default: 3)

Example:
```json
{
  "items": [
    { "item_name": "Product 1", ... },
    { "item_name": "Product 2", ... }
  ],
  "max_workers": 3
}
```

## Grammar Check Details

The Grammar Check endpoint `/api/grammar/check` provides grammar correction for English text using LanguageTool locally. This endpoint works independently and does not require any external services.

### Single Text Check

**Request:**
```json
{
  "text": "This are wrong"
}
```

**Response:**
```json
{
  "original_text": "This are wrong",
  "corrected_text": "This is wrong",
  "has_changes": true,
  "changes_summary": "Changed from 'This are wrong' to 'This is wrong'"
}
```

### Batch Processing

**Request:**
```json
[
  { "text": "This are wrong" },
  { "text": "I has error" }
]
```

**Response:**
```json
[
  {
    "original_text": "This are wrong",
    "corrected_text": "This is wrong",
    "has_changes": true,
    "changes_summary": "Changed from 'This are wrong' to 'This is wrong'"
  },
  {
    "original_text": "I has error",
    "corrected_text": "I have an error",
    "has_changes": true,
    "changes_summary": "Changed from 'I has error' to 'I have an error'"
  }
]
```

### Response Fields

- `original_text`: The input text
- `corrected_text`: The grammar-corrected text
- `has_changes`: Boolean indicating if corrections were made
- `changes_summary`: Summary of changes
- `errors_found`: Number of grammar errors detected

**Note**: This endpoint uses LanguageTool locally and works independently without requiring GradProject service.

## Notes

- `mapping.py` and `ai_att_mapping.py` are **data files** - edit them to add/modify categories and attributes
- `app.py` is the **only server file** - all logic is consolidated here
- `.env` must contain valid API keys for the server to work
- GradProject model (GPU) is **optional** for AI Attributes endpoint:
  - If GradProject is running: Provides image-based color and material detection
  - If GradProject is not running: Shows warning but continues with OpenAI-based attributes extraction
- Batch endpoints use multiprocessing to handle multiple requests simultaneously for better performance
- Grammar check uses LanguageTool locally and works independently without GradProject
