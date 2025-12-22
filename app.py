"""
Unified Flask Server - Category, AI Attributes, and Translation Services
Consolidates three separate Flask servers into a single application with RESTful routes
"""

import os
import sys
import time
import re
from datetime import timedelta
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set environment variables BEFORE importing transformers to avoid chat template lookup
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '0'

from flask import Flask, request, jsonify
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import numpy as np

# OpenAI
from openai import OpenAI

# Torch and Transformers (for GradProject model)
import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer

# Load environment variables
load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
# Flask configuration
FLASK_PORT = 6002
FLASK_HOST = '0.0.0.0'
FLASK_DEBUG = True

# OpenAI configuration (shared by Category and AI Attributes)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = "gpt-4o-mini"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Category service configuration
USE_PHI4 = False  # Set to False to use OpenAI gpt-4o-mini
PHI4_API_URL = os.getenv("API_URL")
PHI4_MODEL_NAME = "phi4:latest"
RATE_LIMIT_DELAY = 3.0  # seconds between API calls (OpenAI rate limiting)

# AI Attributes configuration
HF_TOKEN = os.getenv("HF_TOKEN")
GRAD_SUPPORTED_CATEGORIES = ["fashion", "home and garden"]

# Translation configuration
AYA_API_URL = "http://100.75.237.4:11434/api/generate"
AYA_MODEL_NAME = "aya:8b"

# Import mapping files
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mapping import *

# Import AI attributes mapping
try:
    from openAI_model.ai_att_mapping import category_mapping
except ImportError:
    from ai_att_mapping import category_mapping

# ============================================================
# MONKEY PATCHING FOR GRADPROJECT MODEL
# ============================================================
# Patch HuggingFace to disable chat template lookup
import huggingface_hub
from huggingface_hub.errors import RepositoryNotFoundError

_original_list_repo_tree = huggingface_hub.hf_api.HfApi.list_repo_tree

def patched_list_repo_tree(self, repo_id, *args, **kwargs):
    """Patched version that catches 404 errors for template lookups"""
    try:
        return _original_list_repo_tree(self, repo_id, *args, **kwargs)
    except (RepositoryNotFoundError, Exception) as e:
        error_str = str(e)
        if 'additional_chat_templates' in error_str or '404' in error_str:
            return iter([])  # Return empty iterator
        raise

huggingface_hub.hf_api.HfApi.list_repo_tree = patched_list_repo_tree

# Patch transformers list_repo_templates
from transformers.utils import hub as transformers_hub

_original_list_repo_templates = transformers_hub.list_repo_templates

def patched_list_repo_templates(*args, **kwargs):
    """Patched version that returns empty list to avoid 404 errors"""
    try:
        return _original_list_repo_templates(*args, **kwargs)
    except Exception:
        return []

transformers_hub.list_repo_templates = patched_list_repo_templates

# Patch AutoTokenizer
_original_from_pretrained = AutoTokenizer.from_pretrained

def patched_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
    """Patched version that disables chat template lookup"""
    if 'use_fast' not in kwargs:
        kwargs['use_fast'] = False
    return _original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

AutoTokenizer.from_pretrained = patched_from_pretrained

# ============================================================
# FLASK APP INITIALIZATION
# ============================================================
app = Flask(__name__)

# ============================================================
# MODEL INITIALIZATION - OPENAI CLIENT
# ============================================================
# Initialize OpenAI client if needed
if not USE_PHI4:
    if not OPENAI_API_KEY:
        print("[WARNING] OPENAI_API_KEY not set - OpenAI features will not work")
        openai_client = None
    else:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print(f"[INFO] OpenAI client initialized with API key: {OPENAI_API_KEY[:10]}...")
else:
    openai_client = None

# Track last API call time for rate limiting (category service)
last_api_call_time = 0

# ============================================================
# MODEL INITIALIZATION - GRADPROJECT (LAZY-LOADED)
# ============================================================
grad_model = None
device = None

def load_grad_model():
    """Load GradProject model on GPU (lazy-loaded)"""
    global grad_model, device

    if grad_model is not None:
        print("[INFO] GradProject model already loaded")
        return True

    if not HF_TOKEN:
        print("[WARNING] HF_TOKEN not found - GradProject model disabled")
        return False

    try:
        if not torch.cuda.is_available():
            print("[WARNING] CUDA not available - GradProject model disabled")
            return False

        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

        print(f"[INFO] Loading GradProject model on {torch.cuda.get_device_name(0)}...")

        config = AutoConfig.from_pretrained(
            "Blip-MAE-Botit/BlipMAEModel",
            trust_remote_code=True,
            token=HF_TOKEN,
        )
        config.model_type = "fashion"

        grad_model = AutoModel.from_pretrained(
            "Blip-MAE-Botit/BlipMAEModel",
            trust_remote_code=True,
            config=config,
            token=HF_TOKEN
        ).to(device)

        grad_model.eval()

        # Patch the model's _init_tokenizer method to handle errors gracefully
        original_init_tokenizer = grad_model._init_tokenizer
        tokenizer_initialized = [False]  # Use list to allow modification in closure

        def safe_init_tokenizer():
            """Wrapped tokenizer initialization that catches errors"""
            if tokenizer_initialized[0]:
                return  # Already initialized
            try:
                from transformers import AutoProcessor, BlipImageProcessor
                import transformers
                from transformers import AutoTokenizer as AT

                print("[INFO] Initializing tokenizer...")

                # Load tokenizer with error handling
                try:
                    grad_model.text_tokenizer = AT.from_pretrained(
                        grad_model.tokenizer_repo_id,
                        use_fast=False,
                        trust_remote_code=True,
                        token=HF_TOKEN,
                        local_files_only=False
                    )
                except Exception as e:
                    if '404' in str(e) or 'additional_chat_templates' in str(e):
                        print("[WARNING] Chat template 404, trying local files only...")
                        grad_model.text_tokenizer = AT.from_pretrained(
                            grad_model.tokenizer_repo_id,
                            use_fast=False,
                            trust_remote_code=True,
                            token=HF_TOKEN,
                            local_files_only=True
                        )
                    else:
                        raise

                grad_model.text_tokenizer.add_special_tokens({'bos_token':'[DEC]'})
                grad_model.text_tokenizer.add_special_tokens({"additional_special_tokens":['[ENC]']})
                grad_model.text_tokenizer.enc_token_id = grad_model.text_tokenizer.additional_special_tokens_ids[0]

                # Use BlipImageProcessor directly to avoid missing processor_config.json
                grad_model.image_processor = BlipImageProcessor.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                )
                tokenizer_initialized[0] = True
                print("[INFO] Tokenizer initialized successfully")
            except Exception as e:
                print(f"[ERROR] Failed to initialize tokenizer: {str(e)}")
                raise

        grad_model._init_tokenizer = safe_init_tokenizer

        print("[INFO] GradProject model loaded successfully")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to load GradProject model: {str(e)}")
        print("[INFO] Continuing without GradProject model - will use OpenAI only")
        return False


# ============================================================
# CATEGORY SERVICE - HELPER FUNCTIONS
# ============================================================

def run_model(prompt):
    """
    Run inference using either Phi4 or OpenAI based on USE_PHI4 flag
    Includes rate limiting to respect API limits and error handling
    """
    global last_api_call_time

    # Apply rate limiting (only for OpenAI)
    if not USE_PHI4:
        current_time = time.time()
        time_since_last_call = current_time - last_api_call_time

        if time_since_last_call < RATE_LIMIT_DELAY:
            sleep_time = RATE_LIMIT_DELAY - time_since_last_call
            time.sleep(sleep_time)

        last_api_call_time = time.time()

    try:
        if USE_PHI4:
            # Use local Phi4 model
            payload = {"model": PHI4_MODEL_NAME, "prompt": prompt, "max_tokens": 200, "stream": False}
            r = requests.post(PHI4_API_URL, json=payload)
            r.raise_for_status()
            data = r.json()
            return data["response"].strip()
        else:
            # Use OpenAI gpt-4o-mini
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a strict classification assistant. Follow instructions exactly."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] API call failed: {str(e)}")
        return ""


def classify_shopping_category(item_name, description, item_department, vendor_category):
    prompt = f"""

    You are a strict classification bot.
    Your ONLY job is to return ONE shopping category.

    DO NOT explain. DO NOT add reasoning. DO NOT use multiple lines.

    Item: {item_name}
    Description: {description}
    Item Department: {item_department}
    Vendor Category: {vendor_category}

    Allowed categories:
    {shoppingCategory}

    Return ONLY the category name, nothing else.
    Example valid outputs:
    fashion
    electronics
    groceries

    Now output ONLY the category name:
    """

    result = run_model(prompt)
    print("=============================================")
    print(prompt)
    print("=============================================")
    print(f"[Shopping Category] MODEL RAW RESULT: {result}")

    # clean + normalize
    category = result.lower().replace("'", "").replace('"', "").strip()
    category = category.splitlines()[0].strip()

    # validate
    if category not in shoppingCategory:
        category = ""

    return category


def classify_shopping_subcategory(shopping_category, item_name, description, item_department, vendor_category):
    if not shopping_category or shopping_category not in shoppingSubcategory_map:
        return ""

    subcategory_list = shoppingSubcategory_map[shopping_category]

    # Special handling for stationary category
    if shopping_category == "stationary":
        item_category_mapping = ""
        if shopping_category in itemCategory_map:
            for subcat, items in itemCategory_map[shopping_category].items():
                item_category_mapping += f"\n        - {subcat}: {', '.join(items)}"

        prompt = f"""
        You are a strict classification bot.
        Your ONLY job is to return ONE shopping subcategory.
        DO NOT explain. DO NOT add reasoning. DO NOT use multiple lines.

        Item: {item_name}
        Description: {description}
        Item Department: {item_department}
        Vendor Category: {vendor_category}
        Shopping Category: {shopping_category}

        Allowed subcategories:
        {subcategory_list}

        IMPORTANT MAPPING GUIDANCE FOR STATIONARY ITEMS:
        Use the item type to determine the correct subcategory:
        {item_category_mapping}

        Match the item to its subcategory based on what type of item it is.

        Return ONLY the subcategory name, nothing else.

        Example valid outputs:
        stationary supplies
        office supplies
        arts and crafts

        Now output ONLY the subcategory name:
        """
    else:
        prompt = f"""
        You are a strict classification bot.
        Your ONLY job is to return ONE shopping subcategory.
        DO NOT explain. DO NOT add reasoning. DO NOT use multiple lines.

        Item: {item_name}
        Description: {description}
        Item Department: {item_department}
        Vendor Category: {vendor_category}
        Shopping Category: {shopping_category}

        Allowed subcategories:
        {subcategory_list}

        Return ONLY the subcategory name, nothing else.

        Example valid outputs:
        casual wear
        mobile phones
        bakery

        Now output ONLY the subcategory name:
        """

    result = run_model(prompt)
    print("=============================================")
    print(prompt)
    print("=============================================")
    print(f"[Shopping Subcategory] MODEL RAW RESULT: {result}")

    subcategory = result.lower().replace("'", "").replace('"', "").strip()
    subcategory = subcategory.splitlines()[0].strip()

    if subcategory not in [s.lower() for s in subcategory_list]:
        subcategory = ""

    return subcategory


def classify_item_category(shopping_category, shopping_subcategory, item_name, description, item_department, vendor_category):
    """Classify item into item category"""

    if not shopping_category or not shopping_subcategory:
        return ""

    if shopping_category not in itemCategory_map:
        return ""

    if shopping_subcategory not in itemCategory_map[shopping_category]:
        return ""

    item_category_list = itemCategory_map[shopping_category][shopping_subcategory]

    prompt = f"""
        You are a strict classification bot.
        Your ONLY job is to return ONE item category from the allowed list.

        IMPORTANT: You MUST choose the BEST FIT from the allowed categories below.
        Look at the allowed list carefully and pick the closest match.

        DO NOT explain. DO NOT add reasoning. DO NOT use multiple lines.

        Item: {item_name}
        Description: {description}
        Item Department: {item_department}
        Vendor Category: {vendor_category}
        Shopping Category: {shopping_category}
        Shopping Subcategory: {shopping_subcategory}

        Allowed item categories for {shopping_category} > {shopping_subcategory}:
        {item_category_list}

        Choose the BEST FIT from the list above.
        Return ONLY the category name exactly as it appears in the list, nothing else.

        Example valid outputs:
        t-shirt
        chocolate cake
        smartphone

        Now output ONLY the category name:
        """

    result = run_model(prompt)
    print("=============================================")
    print(prompt)
    print("=============================================")
    print(f"[Item Category] MODEL RAW RESULT: {result}")

    category = result.lower().replace("'", "").replace('"', "").strip()
    category = category.splitlines()[0].strip()

    if category not in [c.lower() for c in item_category_list]:
        category = ""

    return category


def classify_item_subcategory(shopping_category, shopping_subcategory, item_category, item_name, description, item_department, vendor_category):
    """
    Classify item into item subcategory with strict validation.
    Returns tuple: (subcategory, confidence_override)
    """

    if not shopping_category or not shopping_subcategory or not item_category:
        return "", None

    if shopping_category not in itemSubcategory_map:
        return "", None

    if item_category not in itemSubcategory_map[shopping_category]:
        return "", None

    item_subcategory_list = itemSubcategory_map[shopping_category][item_category]

    prompt = f"""
        You are a strict classification bot.
        Your ONLY job is to return ONE item subcategory.
        DO NOT explain. DO NOT add reasoning. DO NOT use multiple lines.

        Item: {item_name}
        Description: {description}
        Item Department: {item_department}
        Vendor Category: {vendor_category}

        Current Classification Path:
        - Shopping Category: {shopping_category}
        - Shopping Subcategory: {shopping_subcategory}
        - Item Category: {item_category}

        Allowed subcategories for {shopping_category} > {item_category}:
        {item_subcategory_list}

        Return ONLY the subcategory name, nothing else.

        Example valid outputs:
        sweatshirt
        running shoes
        vitamin d

        Now output ONLY the subcategory name:
        """

    result = run_model(prompt)
    print("=============================================")
    print(prompt)
    print("=============================================")
    print(f"[Item Subcategory] MODEL RAW RESULT: {result}")
    print("#####################################################################")

    subcategory = result.lower().replace("'", "").replace('"', "").strip()
    subcategory = subcategory.splitlines()[0].strip()

    if subcategory not in [s.lower() for s in item_subcategory_list]:
        print(f"[Item Subcategory] ⚠️  Model returned '{subcategory}' which is not in allowed list. Setting to empty with low confidence.")
        return "", "low"

    return subcategory, None


LOW_CONFIDENCE_ITEM_CATEGORIES = []

def calculate_confidence(vendor_category, shopping_category, item_category=""):
    """
    Calculate confidence level based on:
    1. Whether item_category is in the low confidence list
    2. Whether vendor_category matches shopping_category (case-insensitive)
    """
    if item_category:
        item_cat_normalized = item_category.lower().strip()
        if item_cat_normalized in LOW_CONFIDENCE_ITEM_CATEGORIES:
            return "low"

    if not vendor_category or not shopping_category:
        return "low"

    vendor_cat_normalized = vendor_category.lower().strip()
    shopping_cat_normalized = shopping_category.lower().strip()

    if vendor_cat_normalized == shopping_cat_normalized:
        return "high"
    else:
        return "low"


def classify_single_item(item_data, index):
    """Classify a single item - used for parallel processing"""
    try:
        item_name = item_data.get('item_name', '')
        description = item_data.get('description', '')
        item_department = item_data.get('item_department', '')
        vendor_category = item_data.get('vendor_category', '')

        if not item_name:
            return index, OrderedDict([
                ("shopping_category", ""),
                ("shopping_subcategory", ""),
                ("item_category", ""),
                ("item_subcategory", ""),
                ("confidence", "low")
            ])

        shopping_cat = classify_shopping_category(item_name, description, item_department, vendor_category)
        shopping_subcat = classify_shopping_subcategory(
            shopping_cat, item_name, description, item_department, vendor_category
        )
        item_cat = classify_item_category(
            shopping_cat, shopping_subcat, item_name, description, item_department, vendor_category
        )
        item_subcat, confidence_override = classify_item_subcategory(
            shopping_cat, shopping_subcat, item_cat,
            item_name, description, item_department, vendor_category
        )

        confidence = calculate_confidence(vendor_category, shopping_cat, item_cat)

        if not shopping_cat or not shopping_subcat or not item_cat:
            confidence = "low"

        if confidence_override == "low":
            confidence = "low"

        result = OrderedDict([
            ("shopping_category", shopping_cat),
            ("shopping_subcategory", shopping_subcat),
            ("item_category", item_cat),
            ("item_subcategory", item_subcat),
            ("confidence", confidence)
        ])

        return index, result

    except Exception as e:
        return index, OrderedDict([
            ("shopping_category", ""),
            ("shopping_subcategory", ""),
            ("item_category", ""),
            ("item_subcategory", ""),
            ("confidence", "low")
        ])


# ============================================================
# AI ATTRIBUTES SERVICE - HELPER FUNCTIONS
# ============================================================

def convert_to_python(obj):
    """Convert numpy types to Python types"""
    if isinstance(obj, dict):
        return {k: convert_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_python(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def predict_with_grad_model(images, description, category):
    """Use GradProject model to predict color and material"""
    if grad_model is None:
        return None, None, {}

    try:
        # Transform category to plural form if needed
        category_singular_to_plural = {
            "bag": "bags",
            "top": "tops",
            "dress": "dresses",
            "shoe": "shoes",
            "pant": "pants",
            "skirt": "skirts",
            "jacket": "jackets",
            "coat": "coats",
            "sweater": "sweaters",
            "shirt": "shirts",
            "blouse": "blouses",
        }

        grad_category = category_singular_to_plural.get(category.lower(), category.lower())

        print(f"[DEBUG] Calling grad model with category: {grad_category}, {len(images)} image(s)")

        with torch.inference_mode():
            results = grad_model.generate(
                images_pth=images,
                descriptions=[description] * len(images),
                categories=[grad_category] * len(images),
                attributes=["color", "material"],
                return_confidences=True
            )

        results = convert_to_python(results)

        if not results:
            return None, None, {}

        attributes = results[0][0] if results else {}

        color = None
        material = None
        grad_data = {}

        if "color" in attributes:
            color_data = attributes["color"]
            grad_data["color"] = {
                "value": color_data.get("value"),
                "confidence": float(color_data.get("confidence", 0))
            }
            if grad_data["color"]["confidence"] > 0.5:
                color = grad_data["color"]["value"]
                print(f"[INFO] Grad color: {color} (conf: {grad_data['color']['confidence']:.2f})")

        if "material" in attributes:
            material_data = attributes["material"]
            grad_data["material"] = {
                "value": material_data.get("value"),
                "confidence": float(material_data.get("confidence", 0))
            }
            if grad_data["material"]["confidence"] > 0.5:
                material = grad_data["material"]["value"]
                print(f"[INFO] Grad material: {material} (conf: {grad_data['material']['confidence']:.2f})")

        return color, material, grad_data

    except Exception as e:
        print(f"[ERROR] GradProject prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, {}


def get_attribute_template(shopping_category, item_category):
    """Get template from mapping"""
    category_map = {
        "fashion": "fashion",
        "beauty": "beauty",
        "home and garden": "home_and_garden"
    }

    category_key = category_map.get(shopping_category.lower().strip())

    if category_key in category_mapping:
        item_cat_lower = item_category.lower().strip()
        if item_cat_lower in category_mapping[category_key]:
            return category_mapping[category_key][item_cat_lower][0]

    return None


def run_openai_model(prompt):
    """Run OpenAI model with given prompt"""
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": OPENAI_MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a precise AI attribute extractor for e-commerce products. Follow instructions exactly."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 300,
            "temperature": 0.3
        }

        print(f"[DEBUG] Calling OpenAI API...")
        r = requests.post(OPENAI_API_URL, json=payload, headers=headers)
        r.raise_for_status()
        result = r.json()["choices"][0]["message"]["content"].strip()

        # Clean markdown and placeholders
        result = re.sub(r'```[a-z]*\n?', '', result)
        result = re.sub(r'```\n?', '', result)

        lines = result.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if ':' in line:
                parts = line.split(':', 1)
                attr_name = parts[0].strip()
                attr_value = parts[1].strip() if len(parts) > 1 else ''

                if attr_value.lower() in ['none', 'unknown', 'n/a', 'null', 'empty', 'not specified']:
                    attr_value = ''

                if attr_value:
                    cleaned_lines.append(f"{attr_name}: {attr_value}")
                else:
                    cleaned_lines.append(f"{attr_name}:")
            else:
                if line.lower() not in ['none', 'unknown', 'n/a', 'null', 'empty']:
                    cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    except Exception as e:
        print(f"[ERROR] OpenAI model failed: {str(e)}")
        raise


def extract_ai_attributes(item_name, description, vendor_category, shopping_category,
                         shopping_subcategory, item_category, images=None):
    """
    Extract AI attributes using integrated approach:
    1. If fashion/home&garden with images: use grad model for color/material
    2. Use OpenAI model with color/material hints
    3. Return combined attributes
    """

    # Only extract for allowed categories
    allowed_categories = ["fashion", "beauty", "home and garden"]
    if shopping_category.lower().strip() not in allowed_categories:
        print(f"[DEBUG] Category '{shopping_category}' not allowed")
        return "", {}

    # Get template
    template = get_attribute_template(shopping_category, item_category)
    if not template:
        print(f"[DEBUG] No template found for {shopping_category}/{item_category}")
        return "", {}

    # Step 1: Try grad model for fashion/home&garden with images
    grad_color = None
    grad_material = None
    grad_data = {}

    if (grad_model is not None and
        shopping_category.lower().strip() in GRAD_SUPPORTED_CATEGORIES and
        images and len(images) > 0):

        print(f"[INFO] Using grad model for {shopping_category}...")

        # Lazy-load GradProject model if not already loaded
        if grad_model is None:
            print("[INFO] Lazy-loading GradProject model...")
            load_grad_model()

        if grad_model is not None:
            grad_color, grad_material, grad_data = predict_with_grad_model(
                images,
                description,
                item_category.lower().strip()
            )

    # Step 2: Build OpenAI prompt with grad hints
    input_text = f"""Item Name: {item_name}
Description: {description}
Vendor Category: {vendor_category}
Shopping Category: {shopping_category}
Shopping Subcategory: {shopping_subcategory}
Item Category: {item_category}"""

    # Add color and material hints from grad model
    grad_hints = ""
    if grad_color:
        grad_hints += f"\nDetected Color (from image analysis): {grad_color}"
    if grad_material:
        grad_hints += f"\nDetected Material (from image analysis): {grad_material}"

    prompt = f"""
You are a strict AI attribute extractor for e-commerce products.
Analyze the item below and extract ONLY attributes that can be clearly inferred.
Do NOT guess, do NOT add explanations, do NOT include extra text, do NOT use markdown formatting.
Leave unknown attributes empty.

{input_text}{grad_hints}

INSTRUCTIONS:
- Fill only known attributes; leave others empty
- Use concise English values
- Gender: choose strictly from ["Women", "Men", "Unisex women, Unisex men", "Girls", "Boys", "Unisex girls, unisex boys"]
- Generic Name: identify the main item (e.g. if "Matelda Chocolate cake 120 grams" → Generic Name: "cake")
- Product Name: the product name without size/quantity (e.g. "Matelda Chocolate cake")
- Color: use the detected color from image analysis if provided above
- Material: use the detected material from image analysis if provided above
- Keep the output clean and structured exactly as below
- DO NOT use markdown code blocks (```)
- DO NOT include "None", "unknown", "N/A" - use empty string instead

OUTPUT FORMAT (exactly, no deviations):
{template}

Output ONLY the above format. NO markdown, NO extra lines or explanations.
"""

    result = run_openai_model(prompt)
    return result, grad_data


# ============================================================
# TRANSLATION SERVICE - HELPER FUNCTIONS
# ============================================================

def translate_to_arabic(text: str) -> str:
    """Translate English text to Arabic using Aya model"""
    if not text or text.strip().lower() == "empty":
        return ""

    prompt = (
        "You are a professional English to Arabic translator for e-commerce. "
        "Translate the following text into Arabic. Respond with Arabic text only.\n\n"
        + text
    )

    try:
        payload = {
            "model": AYA_MODEL_NAME,
            "prompt": prompt,
            "max_tokens": 400,
            "stream": False
        }
        response = requests.post(AYA_API_URL, json=payload)
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        print(f"[ERROR] Translation failed: {str(e)}")
        return ""


def translate_batch(texts: list) -> list:
    """Translate multiple texts to Arabic"""
    translations = []
    for text in texts:
        try:
            translations.append(translate_to_arabic(text))
        except Exception as e:
            print(f"[ERROR] Translation error: {e}")
            translations.append("")
    return translations


# ============================================================
# CATEGORY SERVICE - ROUTES
# ============================================================

@app.route('/api/category/classify', methods=['POST'])
def classify_single():
    """Single item classification"""
    try:
        data = request.get_json()

        item_name = data.get('item_name', '')
        description = data.get('description', '')
        item_department = data.get('item_department', '')
        vendor_category = data.get('vendor_category', '')

        if not item_name:
            return jsonify({"error": "item_name is required"}), 400

        # Step 1: Shopping Category
        shopping_cat = classify_shopping_category(item_name, description, item_department, vendor_category)
        if not shopping_cat:
            return jsonify({"error": "shopping_category classification failed - this is a mandatory field"}), 400

        # Step 2: Shopping Subcategory
        shopping_subcat = classify_shopping_subcategory(
            shopping_cat, item_name, description, item_department, vendor_category
        )
        if not shopping_subcat:
            return jsonify({"error": "shopping_subcategory classification failed - this is a mandatory field"}), 400

        # Step 3: Item Category
        item_cat = classify_item_category(
            shopping_cat, shopping_subcat, item_name, description, item_department, vendor_category
        )
        if not item_cat:
            return jsonify({"error": "item_category classification failed - this is a mandatory field"}), 400

        # Step 4: Item Subcategory
        item_subcat, confidence_override = classify_item_subcategory(
            shopping_cat, shopping_subcat, item_cat,
            item_name, description, item_department, vendor_category
        )

        # Calculate confidence
        confidence = calculate_confidence(vendor_category, shopping_cat, item_cat)

        if confidence_override == "low":
            confidence = "low"

        result = OrderedDict([
            ("shopping_category", shopping_cat),
            ("shopping_subcategory", shopping_subcat),
            ("item_category", item_cat),
            ("item_subcategory", item_subcat),
            ("confidence", confidence)
        ])
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/category/classify-csv', methods=['POST'])
def classify_csv():
    """Process CSV from file path"""
    start_time = time.time()

    try:
        data = request.get_json()

        if not data or 'csv_path' not in data:
            return jsonify({"error": "csv_path is required in JSON body"}), 400

        csv_path = data.get('csv_path', '').strip()

        if not csv_path:
            return jsonify({"error": "csv_path cannot be empty"}), 400

        if not os.path.exists(csv_path):
            return jsonify({"error": f"File not found: {csv_path}"}), 404

        # Read CSV file
        df = pd.read_csv(csv_path)

        # Map column names
        column_mapping = {
            'Item (EN)': 'item_name',
            'Description (EN)': 'description',
            'Category/Department (EN)': 'item_department',
            'Vendor Category': 'vendor_category',
            'Variant Name': 'variant_name'
        }

        df.rename(columns=column_mapping, inplace=True)

        if 'item_name' not in df.columns:
            return jsonify({"error": "CSV must contain 'Item (EN)' column"}), 400

        # Add optional columns if they don't exist
        if 'description' not in df.columns:
            df['description'] = ''
        if 'item_department' not in df.columns:
            df['item_department'] = ''
        if 'vendor_category' not in df.columns:
            df['vendor_category'] = ''
        if 'variant_name' not in df.columns:
            df['variant_name'] = ''

        # Initialize result columns
        df['shopping_category'] = ''
        df['shopping_subcategory'] = ''
        df['item_category'] = ''
        df['item_subcategory'] = ''
        df['confidence'] = 'low'

        total_rows = len(df)

        stats = {
            'rows_with_item_name': 0,
            'shopping_category_success': 0,
            'shopping_subcategory_success': 0,
            'item_category_success': 0,
            'item_subcategory_success': 0,
            'fully_classified': 0
        }

        # Generate output filename
        base_name = os.path.basename(csv_path)
        file_name, file_ext = os.path.splitext(base_name)
        output_path = os.path.join(os.path.dirname(csv_path), f"{file_name}_classified{file_ext}")

        print(f"\n[CSV Processing] Starting classification for {total_rows} rows...")
        print(f"[CSV Processing] Creating output file: {output_path}")

        # Write header row
        output_columns = list(df.columns)
        pd.DataFrame(columns=output_columns).to_csv(output_path, index=False)
        print(f"[CSV Processing] Output CSV initialized with headers")

        # Process each row
        for idx, row in tqdm(df.iterrows(), total=total_rows, desc="Classifying items", unit="row"):
            item_name = str(row.get('item_name', '')).strip()
            description = str(row.get('description', '')).strip()
            item_department = str(row.get('item_department', '')).strip()
            vendor_category = str(row.get('vendor_category', '')).strip()

            if not item_name:
                print(f"[Row {idx}] Skipped - No item name")
                pd.DataFrame([row]).to_csv(output_path, mode='a', header=False, index=False)
                continue

            stats['rows_with_item_name'] += 1

            # Classify
            shopping_cat = classify_shopping_category(item_name, description, item_department, vendor_category)
            df.at[idx, 'shopping_category'] = shopping_cat

            if not shopping_cat:
                print(f"[Row {idx}] ⚠️  MANDATORY FIELD FAILED - shopping_category: {item_name}")
                df.at[idx, 'confidence'] = 'low'
                pd.DataFrame([df.loc[idx]]).to_csv(output_path, mode='a', header=False, index=False)
                continue

            stats['shopping_category_success'] += 1

            shopping_subcat = classify_shopping_subcategory(
                shopping_cat, item_name, description, item_department, vendor_category
            )
            df.at[idx, 'shopping_subcategory'] = shopping_subcat

            if not shopping_subcat:
                print(f"[Row {idx}] ⚠️  MANDATORY FIELD FAILED - shopping_subcategory: {item_name}")
                df.at[idx, 'confidence'] = 'low'
                pd.DataFrame([df.loc[idx]]).to_csv(output_path, mode='a', header=False, index=False)
                continue

            stats['shopping_subcategory_success'] += 1

            item_cat = classify_item_category(
                shopping_cat, shopping_subcat, item_name, description, item_department, vendor_category
            )
            df.at[idx, 'item_category'] = item_cat

            if not item_cat:
                print(f"[Row {idx}] ⚠️  MANDATORY FIELD FAILED - item_category: {item_name}")
                df.at[idx, 'confidence'] = 'low'
                pd.DataFrame([df.loc[idx]]).to_csv(output_path, mode='a', header=False, index=False)
                continue

            stats['item_category_success'] += 1

            item_subcat, confidence_override = classify_item_subcategory(
                shopping_cat, shopping_subcat, item_cat,
                item_name, description, item_department, vendor_category
            )
            df.at[idx, 'item_subcategory'] = item_subcat

            confidence = calculate_confidence(vendor_category, shopping_cat, item_cat)

            if confidence_override == "low":
                confidence = "low"

            df.at[idx, 'confidence'] = confidence

            if item_subcat:
                stats['item_subcategory_success'] += 1
                stats['fully_classified'] += 1
            else:
                print(f"[Row {idx}] Failed at item_subcategory: {item_name}")

            pd.DataFrame([df.loc[idx]]).to_csv(output_path, mode='a', header=False, index=False)

        print(f"\n[CSV Processing] Completed!")
        print(f"  Total rows: {total_rows}")
        print(f"  Fully classified: {stats['fully_classified']}")

        end_time = time.time()
        elapsed_seconds = end_time - start_time
        hours, remainder = divmod(int(elapsed_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        time_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        print(f"\n⏱️  Total processing time: {time_formatted}")

        return jsonify({
            "success": True,
            "message": "CSV processed successfully",
            "input_file": csv_path,
            "output_file": output_path,
            "total_rows": total_rows,
            "rows_processed": stats['rows_with_item_name'],
            "processing_time": time_formatted,
            "processing_time_seconds": round(elapsed_seconds, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/category/classify-batch', methods=['POST'])
def classify_batch():
    """Process multiple items in parallel"""
    try:
        start_time = time.time()
        data = request.get_json()

        if not data or 'items' not in data:
            return jsonify({"error": "items array is required"}), 400

        items = data.get('items', [])
        max_workers = data.get('max_workers', 3)

        if not isinstance(items, list):
            return jsonify({"error": "items must be an array"}), 400

        if len(items) == 0:
            return jsonify({"error": "items array cannot be empty"}), 400

        total_items = len(items)
        print(f"\n[Batch Parallel] Processing {total_items} items with {max_workers} workers...")

        results = [None] * total_items

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(classify_single_item, item, idx): idx
                for idx, item in enumerate(items)
            }

            for future in as_completed(future_to_index):
                index, result = future.result()
                results[index] = result

        successful = sum(1 for r in results if r.get('shopping_category') and r.get('shopping_subcategory') and r.get('item_category'))
        failed = total_items - successful

        end_time = time.time()
        elapsed_seconds = end_time - start_time
        hours, remainder = divmod(int(elapsed_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        time_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        print(f"[Batch Parallel] Completed {total_items} items in {time_formatted}")
        print(f"[Batch Parallel] Success: {successful}/{total_items} | Failed: {failed}/{total_items}")

        return jsonify({
            "success": True,
            "results": results,
            "total_items": total_items,
            "successful_classifications": successful,
            "failed_classifications": failed,
            "processing_time": time_formatted,
            "processing_time_seconds": round(elapsed_seconds, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# AI ATTRIBUTES SERVICE - ROUTES
# ============================================================

@app.route('/api/attributes/extract', methods=['POST'])
def extract_attributes():
    """Extract AI attributes"""
    try:
        data = request.get_json()

        item_name = data.get('item_name', '')
        description = data.get('description', '')
        vendor_category = data.get('vendor_category', '')
        shopping_category = data.get('shopping_category', '')
        shopping_subcategory = data.get('shopping_subcategory', '')
        item_category = data.get('item_category', '')
        images = data.get('images', [])

        if not all([item_name, shopping_category, item_category]):
            return jsonify({
                "error": "item_name, shopping_category, and item_category are required"
            }), 400

        allowed_categories = ["fashion", "beauty", "home and garden"]
        if shopping_category.lower().strip() not in allowed_categories:
            return jsonify({
                "success": False,
                "message": f"Category '{shopping_category}' not allowed. Allowed: {', '.join(allowed_categories)}",
                "ai_attributes": ""
            }), 200

        # Extract attributes
        attributes, grad_data = extract_ai_attributes(
            item_name, description, vendor_category,
            shopping_category, shopping_subcategory, item_category,
            images
        )

        return jsonify({
            "success": True,
            "message": "AI attributes extracted successfully.",
            "ai_attributes": attributes,
            "ai_attributes_array": attributes.split('\n') if attributes else [],
            "grad_model_used": bool(grad_data),
            "grad_predictions": grad_data if grad_data else None
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/attributes/health', methods=['GET'])
def attributes_health():
    """AI Attributes health check"""
    return jsonify({
        "status": "healthy",
        "service": "AI Attributes Service",
        "grad_model_loaded": grad_model is not None,
        "openai_configured": bool(OPENAI_API_KEY),
        "gpu_available": torch.cuda.is_available()
    })


# ============================================================
# TRANSLATION SERVICE - ROUTES
# ============================================================

@app.route('/api/translation/translate', methods=['POST'])
def translate_endpoint():
    """
    Translate English to Arabic (single or batch)

    Input JSON (single):
    { "text": "Your text here" }

    Input JSON (batch):
    { "texts": ["Text 1", "Text 2"] }
    """
    try:
        data = request.get_json(force=True)

        if "texts" in data:
            if not isinstance(data["texts"], list):
                return jsonify({"error": "'texts' must be a list"}), 400
            return jsonify({
                "translations": translate_batch(data["texts"])
            })

        if "text" in data:
            return jsonify({
                "translation": translate_to_arabic(data["text"])
            })

        return jsonify({"error": "Provide 'text' or 'texts'"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/translation/health', methods=['GET'])
def translation_health():
    """Translation health check"""
    return jsonify({
        "status": "healthy",
        "service": "Arabic Translation Service"
    })


# ============================================================
# PIPELINE SERVICE - ROUTES (NEW)
# ============================================================

@app.route('/api/pipeline/process', methods=['POST'])
def pipeline_process():
    """
    End-to-end processing: classification → attributes → translation

    Input:
    {
        "item_name": "...",
        "description": "...",
        "item_department": "...",
        "vendor_category": "...",
        "images": ["..."],
        "translate": true  // optional, default false
    }
    """
    try:
        data = request.get_json()

        item_name = data.get('item_name', '')
        description = data.get('description', '')
        item_department = data.get('item_department', '')
        vendor_category = data.get('vendor_category', '')
        images = data.get('images', [])
        should_translate = data.get('translate', False)

        if not item_name:
            return jsonify({"error": "item_name is required"}), 400

        # Step 1: Classification
        shopping_cat = classify_shopping_category(item_name, description, item_department, vendor_category)
        if not shopping_cat:
            return jsonify({"error": "Classification failed at shopping_category"}), 400

        shopping_subcat = classify_shopping_subcategory(
            shopping_cat, item_name, description, item_department, vendor_category
        )
        if not shopping_subcat:
            return jsonify({"error": "Classification failed at shopping_subcategory"}), 400

        item_cat = classify_item_category(
            shopping_cat, shopping_subcat, item_name, description, item_department, vendor_category
        )
        if not item_cat:
            return jsonify({"error": "Classification failed at item_category"}), 400

        item_subcat, confidence_override = classify_item_subcategory(
            shopping_cat, shopping_subcat, item_cat,
            item_name, description, item_department, vendor_category
        )

        confidence = calculate_confidence(vendor_category, shopping_cat, item_cat)
        if confidence_override == "low":
            confidence = "low"

        classification_result = OrderedDict([
            ("shopping_category", shopping_cat),
            ("shopping_subcategory", shopping_subcat),
            ("item_category", item_cat),
            ("item_subcategory", item_subcat),
            ("confidence", confidence)
        ])

        # Step 2: AI Attributes Extraction
        ai_attributes_result = {}
        allowed_categories = ["fashion", "beauty", "home and garden"]

        if shopping_cat.lower().strip() in allowed_categories:
            attributes, grad_data = extract_ai_attributes(
                item_name, description, vendor_category,
                shopping_cat, shopping_subcat, item_cat,
                images
            )

            ai_attributes_result["english"] = attributes
            ai_attributes_result["grad_model_used"] = bool(grad_data)
            ai_attributes_result["grad_predictions"] = grad_data if grad_data else None

            # Step 3: Translation (if requested)
            if should_translate and attributes:
                arabic_translation = translate_to_arabic(attributes)
                ai_attributes_result["arabic"] = arabic_translation
        else:
            ai_attributes_result["message"] = f"Category '{shopping_cat}' not supported for AI attributes"

        return jsonify({
            "success": True,
            "classification": classification_result,
            "ai_attributes": ai_attributes_result
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/pipeline/process-batch', methods=['POST'])
def pipeline_process_batch():
    """
    Batch version of pipeline processing

    Input:
    {
        "items": [
            {
                "item_name": "...",
                "description": "...",
                "item_department": "...",
                "vendor_category": "...",
                "images": ["..."]
            }
        ],
        "translate": true  // optional, default false
    }
    """
    try:
        data = request.get_json()

        if not data or 'items' not in data:
            return jsonify({"error": "items array is required"}), 400

        items = data.get('items', [])
        should_translate = data.get('translate', False)

        if not isinstance(items, list):
            return jsonify({"error": "items must be an array"}), 400

        if len(items) == 0:
            return jsonify({"error": "items array cannot be empty"}), 400

        results = []

        for idx, item in enumerate(items):
            try:
                # Process each item through the pipeline
                item['translate'] = should_translate

                # Use the single process endpoint logic
                with app.test_request_context(
                    path='/api/pipeline/process',
                    method='POST',
                    json=item
                ):
                    response = pipeline_process()
                    if isinstance(response, tuple):
                        result_data = response[0].get_json()
                    else:
                        result_data = response.get_json()

                    results.append(result_data)
            except Exception as e:
                results.append({
                    "success": False,
                    "error": str(e)
                })

        successful = sum(1 for r in results if r.get('success', False))
        failed = len(results) - successful

        return jsonify({
            "success": True,
            "results": results,
            "total_items": len(items),
            "successful": successful,
            "failed": failed
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# GLOBAL HEALTH ENDPOINT
# ============================================================

@app.route('/health', methods=['GET'])
def global_health():
    """Global health check for all services"""
    return jsonify({
        "status": "healthy",
        "services": ["category", "ai_attributes", "translation", "pipeline"],
        "models": {
            "openai": bool(OPENAI_API_KEY),
            "gradproject_loaded": grad_model is not None,
            "ollama": True
        },
        "port": FLASK_PORT
    })


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("UNIFIED FLASK SERVER")
    print("="*70)

    # Display configuration
    print("\nConfiguration:")
    if USE_PHI4:
        print(f"  Category Model: Phi4 (Local)")
        print(f"  Phi4 API URL: {PHI4_API_URL}")
    else:
        print(f"  Category Model: OpenAI gpt-4o-mini")

    print(f"  OpenAI API Key: {'[OK] Configured' if OPENAI_API_KEY else '[X] Missing'}")
    print(f"  HF Token: {'[OK] Configured' if HF_TOKEN else '[X] Missing (GradProject disabled)'}")
    print(f"  GPU: {'[OK] Available' if torch.cuda.is_available() else '[X] Not available'}")

    print("\nEndpoints:")
    print("  Category Service:")
    print("    POST /api/category/classify         - Single item classification")
    print("    POST /api/category/classify-csv     - Process CSV file")
    print("    POST /api/category/classify-batch   - Parallel batch classification")

    print("\n  AI Attributes Service:")
    print("    POST /api/attributes/extract        - Extract AI attributes")
    print("    GET  /api/attributes/health         - Health check")

    print("\n  Translation Service:")
    print("    POST /api/translation/translate     - English → Arabic translation")
    print("    GET  /api/translation/health        - Health check")

    print("\n  Pipeline Service:")
    print("    POST /api/pipeline/process          - End-to-end processing")
    print("    POST /api/pipeline/process-batch    - Batch pipeline processing")

    print("\n  Global:")
    print("    GET  /health                        - Global health check")

    print("\n" + "="*70)
    print(f"Starting unified server on http://{FLASK_HOST}:{FLASK_PORT}")
    print("="*70 + "\n")

    # Note: GradProject model will be lazy-loaded on first use
    print("[INFO] GradProject model will be lazy-loaded on first /api/attributes/extract request")
    print("")

    app.run(debug=FLASK_DEBUG, host=FLASK_HOST, port=FLASK_PORT)
