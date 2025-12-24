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

## How It Works

```
app.py
  ├── Imports mapping.py          → Category hierarchies
  ├── Imports ai_att_mapping.py   → Attribute templates
  └── Uses .env                    → API keys

User Request → app.py → Uses mappings + API keys → Returns result
```

## File Dependencies

| File | Depends On |
|------|------------|
| `app.py` | `mapping.py`, `ai_att_mapping.py`, `.env`, `requirements.txt` |
| `mapping.py` | None (standalone data) |
| `ai_att_mapping.py` | None (standalone data) |
| `requirements.txt` | None (package list) |
| `.env` | None (configuration) |

## Notes

- `mapping.py` and `ai_att_mapping.py` are **data files** - edit them to add/modify categories and attributes
- `app.py` is the **only server file** - all logic is consolidated here
- `.env` must contain valid API keys for the server to work
- GradProject model (GPU) is **optional** and lazy-loaded on first use
