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

from flask import Flask, request, jsonify
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import language_tool_python

# OpenAI
from openai import OpenAI

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
GRAD_SUPPORTED_CATEGORIES = ["fashion", "home and garden"]
GRADPROJECT_API_URL = os.getenv("GRADPROJECT_API_URL", "http://localhost:5000/predict")  # External GradProject service
USE_OPENAI_FALLBACK = True  # If True, fall back to OpenAI for color/material when GradProject fails or isn't used
USE_GPT4O_VISION = False  # If False, skip GPT-4o vision (quota/cost control). Text-only extraction still works.

# Translation configuration
AYA_API_URL = os.getenv("AYA_API_URL", "http://localhost:11434/api/generate")
AYA_MODEL_NAME = "aya:8b"

# Import mapping files
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mapping import *

# Import AI attributes mapping
from ai_att_mapping import category_mapping

# Import standardization mapping
from standardization_mapping import standardization_map
from standardization_synonyms import synonyms_map

# Flatten synonyms_map ({canonical: [synonyms]}) into a fast reverse lookup
# of shape { attribute_key: { synonym_lowercase: canonical_value } } so each
# attribute lookup stays O(1) at request time.
_SYNONYM_LOOKUP = {}
for _attr, _canonical_to_syns in synonyms_map.items():
    _bucket = {}
    for _canonical, _syns in _canonical_to_syns.items():
        # Allow the canonical value itself to resolve via this map too
        _bucket[_canonical.strip().lower()] = _canonical
        for _syn in _syns:
            _bucket[_syn.strip().lower()] = _canonical
    _SYNONYM_LOOKUP[_attr] = _bucket

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
# GRAMMAR CHECK - LANGUAGE TOOL INITIALIZATION
# ============================================================
# Initialize LanguageTool for grammar checking (lazy loading)
grammar_tool = None

def get_grammar_tool():
    """Lazy load grammar tool on first use"""
    global grammar_tool
    if grammar_tool is None:
        print("[INFO] Initializing LanguageTool for grammar checking...")
        grammar_tool = language_tool_python.LanguageTool('en-US')
        print("[INFO] LanguageTool initialized successfully")
    return grammar_tool

# ============================================================
# GRADPROJECT SERVICE - EXTERNAL API
# ============================================================
# GradProject runs as a separate service on port 5000
# No model loading needed - just make HTTP requests


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
    desc_note = description.strip() if description and description.strip() else "[not provided]"
    dept_note = item_department.strip() if item_department and item_department.strip() else "[not provided]"
    vcat_note = vendor_category.strip() if vendor_category and vendor_category.strip() else "[not provided]"

    prompt = f"""You are a strict classification bot.
Your ONLY job is to return ONE shopping category from the allowed list.
DO NOT explain. DO NOT add reasoning. DO NOT use multiple lines.

Item: {item_name}
Description: {desc_note}
Item Department: {dept_note}
Vendor Category: {vcat_note}

Allowed categories:
{shoppingCategory}

DISAMBIGUATION RULES — read carefully before choosing:
- Any product intended for animals/pets (shampoo, food, toys, supplements, accessories, leashes, litter) → "pet care", NOT "pharmacy", "beauty", or "health and nutrition"
- Ready-to-eat meals, restaurant dishes, takeaway food, cafe drinks → "restaurants", NOT "groceries"
- Packaged/raw food, supermarket goods, household food items, cooking ingredients → "groceries", NOT "restaurants"
- Human skincare, makeup, haircare, perfume, cosmetics → "beauty", NOT "pharmacies"
- Medicine, prescription drugs, medical devices, wound care, incontinence products → "pharmacies", NOT "beauty" or "health and nutrition"
- Human vitamins, protein powder, dietary supplements, sports nutrition → "health and nutrition", NOT "pharmacies"
- Baby/toddler clothing → "fashion", NOT "kids"
- Baby furniture, strollers, baby monitors, toys, baby care products → "kids", NOT "fashion"
- If Description and Vendor Category are both [not provided], rely on the item name alone and choose the most specific match.

EXAMPLES:
Item: Pet Shampoo → pet care
Item: Dog Supplements → pet care
Item: Cat Food → pet care
Item: Grilled Chicken Meal → restaurants
Item: Raw Chicken Breast → groceries
Item: Pasta (uncooked, packaged) → groceries
Item: Vitamin C Tablets (for humans) → health and nutrition
Item: Amoxicillin / Paracetamol → pharmacies
Item: Baby Diaper → pharmacies
Item: Face Cream / Lipstick / Mascara → beauty
Item: Baby Dress → fashion
Item: Baby Stroller → kids

Return ONLY the category name, nothing else.
Now output ONLY the category name:"""

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

    desc_note = description.strip() if description and description.strip() else "[not provided]"
    dept_note = item_department.strip() if item_department and item_department.strip() else "[not provided]"
    vcat_note = vendor_category.strip() if vendor_category and vendor_category.strip() else "[not provided]"

    # Special handling for stationary category
    if shopping_category == "stationary":
        item_category_mapping = ""
        if shopping_category in itemCategory_map:
            for subcat, items in itemCategory_map[shopping_category].items():
                item_category_mapping += f"\n        - {subcat}: {', '.join(items)}"

        prompt = f"""You are a strict classification bot.
Your ONLY job is to return ONE shopping subcategory.
DO NOT explain. DO NOT add reasoning. DO NOT use multiple lines.

Item: {item_name}
Description: {desc_note}
Item Department: {dept_note}
Vendor Category: {vcat_note}
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

Now output ONLY the subcategory name:"""
    else:
        # Build category-specific disambiguation hints
        disambiguation = ""
        if shopping_category == "pharmacies":
            disambiguation = """
DISAMBIGUATION: Within pharmacies subcategories:
- Products marketed for women's hygiene, menstrual care → "women's care"
- Products for men's hygiene or men-specific health → "men's care"
- Contact lenses, eye drops, eyeglasses accessories → "eye care"
- Toothbrush, toothpaste, mouthwash, dental floss → "dental care"
- Adult diapers, incontinence pads → "incontinence"
- Prescription drugs, OTC medicine, pain relief, fever → "medicine"
- Bandages, antiseptics, blood pressure monitors, thermometers → "first aid and medical equipment"
"""
        elif shopping_category == "fashion":
            disambiguation = """
DISAMBIGUATION: Within fashion subcategories:
- Everyday clothing (t-shirts, jeans, dresses) → "casual wear"
- Sports clothing (tracksuits, leggings, jerseys) → "sportswear"
- Shoes, sandals, boots, slippers → "footwear"
- Rings, necklaces, earrings, bracelets → "jewelry"
- Sunglasses, eyeglasses frames → "eyewear"
- Belts, bags, wallets, hats, scarves → "accessories"
- Bras, underwear, boxers → "undergarments"
- Pajamas, nightgowns → "sleepwear"
- Baby/toddler clothing → still classify under the relevant fashion subcategory (e.g. casual wear, baby clothes)
"""
        elif shopping_category == "health and nutrition":
            disambiguation = """
DISAMBIGUATION: Within health and nutrition subcategories:
- Vitamin A/C/D/B12/multivitamins → "vitamins"
- Whey protein, casein, plant protein → "protein products"
- Pre-workout, creatine, BCAAs, post-workout → "preformance supplements"
- General supplements not fitting above → "dietary supplements"
- Herbal remedies, natural oils, plant extracts → "natural solutions"
- Weight loss pills, fat burners, meal replacements → "weight management"
"""

        prompt = f"""You are a strict classification bot.
Your ONLY job is to return ONE shopping subcategory.
DO NOT explain. DO NOT add reasoning. DO NOT use multiple lines.

Item: {item_name}
Description: {desc_note}
Item Department: {dept_note}
Vendor Category: {vcat_note}
Shopping Category: {shopping_category}

Allowed subcategories:
{subcategory_list}
{disambiguation}
Return ONLY the subcategory name, nothing else.
Example valid outputs:
casual wear
footwear
vitamins

Now output ONLY the subcategory name:"""

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

    desc_note = description.strip() if description and description.strip() else "[not provided]"
    dept_note = item_department.strip() if item_department and item_department.strip() else "[not provided]"
    vcat_note = vendor_category.strip() if vendor_category and vendor_category.strip() else "[not provided]"

    prompt = f"""You are a strict classification bot.
Your ONLY job is to return ONE item category from the allowed list below.
DO NOT explain. DO NOT add reasoning. DO NOT use multiple lines.

Item: {item_name}
Description: {desc_note}
Item Department: {dept_note}
Vendor Category: {vcat_note}
Shopping Category: {shopping_category}
Shopping Subcategory: {shopping_subcategory}

Allowed item categories for {shopping_category} > {shopping_subcategory}:
{item_category_list}

RULES:
- You MUST pick from the allowed list above — do NOT invent a new name.
- Choose the most specific and accurate match based on what the item physically is.
- If the item name and description conflict, trust the item name.
- If description is [not provided], rely on the item name and vendor category.

Return ONLY the category name exactly as it appears in the list, nothing else.
Now output ONLY the category name:"""

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

    desc_note = description.strip() if description and description.strip() else "[not provided]"
    dept_note = item_department.strip() if item_department and item_department.strip() else "[not provided]"
    vcat_note = vendor_category.strip() if vendor_category and vendor_category.strip() else "[not provided]"

    prompt = f"""You are a strict classification bot.
Your ONLY job is to return ONE item subcategory from the allowed list below.
DO NOT explain. DO NOT add reasoning. DO NOT use multiple lines.

Item: {item_name}
Description: {desc_note}
Item Department: {dept_note}
Vendor Category: {vcat_note}

Current Classification Path:
- Shopping Category: {shopping_category}
- Shopping Subcategory: {shopping_subcategory}
- Item Category: {item_category}

Allowed subcategories for {shopping_category} > {item_category}:
{item_subcategory_list}

RULES:
- You MUST pick from the allowed list above — do NOT invent a new name.
- Choose the closest match to what the item physically is.
- If description is [not provided], rely solely on the item name and the classification path above.

Return ONLY the subcategory name exactly as it appears in the list, nothing else.
Now output ONLY the subcategory name:"""

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

def calculate_confidence(vendor_category, shopping_category, item_category="", description=""):
    """
    Calculate confidence level based on:
    1. Whether item_category is in the low confidence list
    2. Whether description AND vendor_category are both missing (unreliable inputs)
    3. Whether vendor_category matches shopping_category (case-insensitive)
    """
    if item_category:
        item_cat_normalized = item_category.lower().strip()
        if item_cat_normalized in LOW_CONFIDENCE_ITEM_CATEGORIES:
            return "low"

    if not vendor_category or not shopping_category:
        return "low"

    # Force low confidence when both primary context signals are absent
    desc_missing = not description or not description.strip()
    vcat_vague = len(vendor_category.strip()) < 3
    if desc_missing and vcat_vague:
        return "low"

    vendor_cat_normalized = vendor_category.lower().strip()
    shopping_cat_normalized = shopping_category.lower().strip()

    if vendor_cat_normalized == shopping_cat_normalized:
        return "high"
    else:
        return "low"


def classify_single_item(item_data, index, shopping_category_override=None):
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

        if shopping_category_override and shopping_category_override.lower().strip() in shoppingCategory:
            shopping_cat = shopping_category_override.lower().strip()
        else:
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

        confidence = calculate_confidence(vendor_category, shopping_cat, item_cat, description)

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

def predict_with_grad_model(images, description, category, shopping_category="fashion"):
    """Call external GradProject service to predict color and material"""
    if not images or len(images) == 0:
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

        print(f"[DEBUG] Calling external GradProject service with category: {grad_category}, domain: {shopping_category}, {len(images)} image(s)")

        # Make HTTP request to external GradProject service
        payload = {
            "images": images,
            "descriptions": [description],
            "categories": [grad_category],
            "attributes": ["color", "material"],
            "domain": shopping_category
        }

        response = requests.post(GRADPROJECT_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        # Extract attributes from response
        attributes = result.get("attributes", {})

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

    except requests.exceptions.ConnectionError:
        print(f"[WARNING] Cannot connect to GradProject service at {GRADPROJECT_API_URL}")
        print("[INFO] Continuing without GradProject predictions...")
        return None, None, {"warning": "GradProject service unavailable - predictions not included"}
    except Exception as e:
        print(f"[WARNING] GradProject prediction failed: {str(e)}")
        print("[INFO] Continuing without GradProject predictions...")
        return None, None, {"warning": f"GradProject prediction failed: {str(e)}"}


def predict_color_material_openai(item_name, description, item_category, images=None):
    """
    Fallback: use OpenAI GPT-4o vision to predict color and material from product images.
    If no images provided, falls back to text-only analysis.
    Returns (color, material, confidence_data) where confidence_data contains confidence scores.
    """
    # Hard kill switch: skip vision entirely (e.g. quota exhausted)
    if not USE_GPT4O_VISION:
        print("[INFO] USE_GPT4O_VISION=False — skipping GPT-4o color/material vision call.")
        return None, None, {"skipped": "USE_GPT4O_VISION=False"}

    try:
        VISION_MODEL = "gpt-4o"

        prompt_text = f"""You are a product attribute extractor. Analyze the product and identify the color and material.

Item Name: {item_name}
Description: {description}
Item Category: {item_category}

Respond in EXACTLY this format (one per line, no extra text):
color: <color>
color_confidence: <0.0 to 1.0>
material: <material>
material_confidence: <0.0 to 1.0>

Rules:
- For color: identify the dominant/primary color of the product
- For material: identify the main material (e.g. cotton, leather, plastic, wood, metal, polyester, silk, etc.)
- Confidence: 1.0 = very certain, 0.5 = somewhat certain, below 0.3 = guessing
- If you cannot determine a value, leave it empty after the colon and set confidence to 0.0"""

        # Build messages with image content if available
        user_content = []

        if images and len(images) > 0:
            # Add images for GPT-4o vision analysis
            for img_url in images[:3]:  # Limit to 3 images to control costs
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": img_url, "detail": "low"}
                })
            print(f"[INFO] OpenAI Vision fallback - analyzing {min(len(images), 3)} image(s) with {VISION_MODEL}")
        else:
            print(f"[INFO] OpenAI Vision fallback - no images, using text-only with {VISION_MODEL}")

        user_content.append({"type": "text", "text": prompt_text})

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": VISION_MODEL,
            "messages": [
                {"role": "system", "content": "You are a precise product attribute extractor. When images are provided, analyze them carefully to determine color and material."},
                {"role": "user", "content": user_content}
            ],
            "max_tokens": 100,
            "temperature": 0.1
        }

        r = requests.post(OPENAI_API_URL, json=payload, headers=headers)
        r.raise_for_status()
        result = r.json()["choices"][0]["message"]["content"].strip()

        color = None
        material = None
        color_confidence = 0.0
        material_confidence = 0.0

        for line in result.split('\n'):
            line = line.strip().lower()
            if line.startswith('color_confidence:'):
                val = line.split(':', 1)[1].strip()
                try:
                    color_confidence = float(val)
                except ValueError:
                    color_confidence = 0.0
            elif line.startswith('color:'):
                val = line.split(':', 1)[1].strip()
                if val and val not in ['none', 'unknown', 'n/a', 'empty', '']:
                    color = val
            elif line.startswith('material_confidence:'):
                val = line.split(':', 1)[1].strip()
                try:
                    material_confidence = float(val)
                except ValueError:
                    material_confidence = 0.0
            elif line.startswith('material:'):
                val = line.split(':', 1)[1].strip()
                if val and val not in ['none', 'unknown', 'n/a', 'empty', '']:
                    material = val

        confidence_data = {
            "color_confidence": color_confidence,
            "material_confidence": material_confidence,
            "model": VISION_MODEL,
            "images_analyzed": min(len(images), 3) if images else 0
        }

        print(f"[INFO] OpenAI Vision fallback - color: {color} (conf: {color_confidence}), material: {material} (conf: {material_confidence})")
        return color, material, confidence_data

    except Exception as e:
        print(f"[WARNING] OpenAI Vision fallback for color/material failed: {str(e)}")
        return None, None, {}


_HOME_GARDEN_SUBCAT_NORMALIZE = {
    "kitchenwear": "kitchenware",
    "household products": "household products / Home Scent",
    "storage and organization": "storage & organization",
    "bed and bath": "bed & bath",
    "gardening and outdoor": "gardening & outdoor",
    "hardware and home improvement": "hardware & home improvement",
}

# Defines which field is used to look up the attribute template for each category.
# fashion / electronics / pet_care: mapping keys are item types → use item_category
# all other categories: mapping keys are subcategory names → use shopping_subcategory
_TEMPLATE_LOOKUP_FIELD = {
    "fashion": "item_category",  # (item_category)
    "electronics": "item_category", # (item_category)
    "pet_care": "item_category", # (item_category)
    
    "beauty": "shopping_subcategory",
    "home_and_garden": "shopping_subcategory",
    "kids": "shopping_subcategory",
    "sports": "shopping_subcategory",
    "flowers_and_gifts": "shopping_subcategory",
    "stationary": "shopping_subcategory",
    "automotive": "shopping_subcategory",
    "groceries": "shopping_subcategory",
    "health_and_nutrition": "shopping_subcategory",
    "pharmacies": "shopping_subcategory",
    "entertainment": "shopping_subcategory",
}

# Normalizes the raw shopping_category string to the key used in category_mapping.
_CATEGORY_KEY_MAP = {
    "fashion": "fashion",
    "beauty": "beauty",
    
    "home and garden": "home_and_garden",
    "home_and_garden": "home_and_garden",
    
    "kids": "kids",
    "sports": "sports",
    
    "flowers and gifts": "flowers_and_gifts",
    "flowers_and_gifts": "flowers_and_gifts",
    
    "stationary": "stationary",
    "stationery": "stationary",
    
    "automotive": "automotive",
    "groceries": "groceries",
    
    "pet care": "pet_care",
    "pet_care": "pet_care",
    
    "health and nutrition": "health_and_nutrition",
    "health_and_nutrition": "health_and_nutrition",
    
    "pharmacies": "pharmacies",
    "electronics": "electronics",
    "entertainment": "entertainment",
}


def get_attribute_template(shopping_category, item_category, shopping_subcategory=""):
    """Get attribute template from mapping.

    The lookup key is chosen per-category via _TEMPLATE_LOOKUP_FIELD:
    - fashion, electronics, pet_care → item_category
    - all other categories            → shopping_subcategory
    """
    category_key = _CATEGORY_KEY_MAP.get(shopping_category.lower().strip())

    if category_key not in category_mapping:
        return None

    mapping = category_mapping[category_key]
    lookup_field = _TEMPLATE_LOOKUP_FIELD.get(category_key, "shopping_subcategory")

    if lookup_field == "item_category":
        lookup_key = item_category.lower().strip()
    else:
        lookup_key = shopping_subcategory.lower().strip()
        lookup_key = _HOME_GARDEN_SUBCAT_NORMALIZE.get(lookup_key, lookup_key)

    # Case-insensitive match
    mapping_lower = {k.lower(): v for k, v in mapping.items()}
    return mapping_lower[lookup_key][0] if lookup_key in mapping_lower else None


def _normalize_image_urls(images, limit=4):
    """Accept either list of URL strings or list of dicts ({"large":..., "medium":..., "small":...})
    and return a flat list of URL strings (preferring 'large').
    """
    if not images:
        return []
    urls = []
    for img in images:
        if isinstance(img, str):
            urls.append(img)
        elif isinstance(img, dict):
            url = img.get("large") or img.get("medium") or img.get("small") or img.get("url")
            if url:
                urls.append(url)
        if len(urls) >= limit:
            break
    return urls


def run_openai_model(prompt, images=None):
    """Run OpenAI model with given prompt.

    If `images` is provided (list of URLs or image dicts), automatically switches
    to the gpt-4o vision model so the model can analyze the product photos.
    Otherwise uses the configured text model.
    """
    try:
        image_urls = _normalize_image_urls(images, limit=4) if images else []
        # Respect the global vision kill-switch — fall back to text-only if disabled.
        use_vision = bool(image_urls) and USE_GPT4O_VISION
        if image_urls and not USE_GPT4O_VISION:
            print("[INFO] USE_GPT4O_VISION=False — ignoring images, using text-only extraction.")
        model = "gpt-4o" if use_vision else OPENAI_MODEL_NAME

        if use_vision:
            user_content = []
            for url in image_urls:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": url, "detail": "low"}
                })
            user_content.append({"type": "text", "text": prompt})
            user_message = {"role": "user", "content": user_content}
            print(f"[DEBUG] Calling OpenAI Vision ({model}) with {len(image_urls)} image(s)...")
        else:
            user_message = {"role": "user", "content": prompt}
            print(f"[DEBUG] Calling OpenAI API ({model}) text-only...")

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a precise AI attribute extractor for e-commerce products. Follow instructions exactly. When images are provided, analyze them carefully to determine color, material, pattern, and other visual attributes."},
                user_message
            ],
            "max_tokens": 300,
            "temperature": 0.3
        }
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


def merge_with_template(template, model_output):
    """Reconstruct output using the template's attribute order.

    Any attribute the model skipped or left out is included with an empty value,
    ensuring the full template structure is always returned.
    """
    # Build a lookup from lowercased attribute name → value from model output
    model_values = {}
    for line in model_output.split('\n'):
        line = line.strip()
        if ':' in line:
            name, _, value = line.partition(':')
            model_values[name.strip().lower()] = value.strip()

    # Rebuild in template order, filling missing attrs with empty string
    result_lines = []
    for line in template.split('\n'):
        line = line.strip()
        if not line or ':' not in line:
            continue
        name, _, _ = line.partition(':')
        attr_name = name.strip()
        value = model_values.get(attr_name.lower(), '')
        result_lines.append(f"{attr_name}: {value}" if value else f"{attr_name}:")

    return '\n'.join(result_lines)


def extract_ai_attributes(item_name, description, vendor_category, shopping_category,
                         shopping_subcategory, item_category, images=None, variant_name=''):
    """
    Extract AI attributes using integrated approach:
    1. If fashion/home&garden with images: use grad model for color/material
    2. Use OpenAI model (vision-enabled when images are present) for full extraction
    3. Return combined attributes
    """

    # Get template
    template = get_attribute_template(shopping_category, item_category, shopping_subcategory)
    if not template:
        print(f"[DEBUG] No template found for {shopping_category}/{item_category}")
        return "", {}

    # Normalize images: accept either raw URL strings or dicts with large/medium/small.
    image_urls = _normalize_image_urls(images, limit=4)

    # Step 1: Try grad model for fashion/home&garden with images
    grad_color = None
    grad_material = None
    grad_data = {}
    used_fallback = False

    if (shopping_category.lower().strip() in GRAD_SUPPORTED_CATEGORIES and
        image_urls):

        print(f"[INFO] Calling external GradProject service for {shopping_category}...")
        grad_color, grad_material, grad_data = predict_with_grad_model(
            image_urls,
            description,
            item_category.lower().strip(),
            shopping_category.lower().strip()
        )

    # Step 1b: Fallback to OpenAI GPT-4o Vision if GradProject didn't produce results
    # Only relevant for categories the grad model supports (fashion, home and garden)
    if shopping_category.lower().strip() in GRAD_SUPPORTED_CATEGORIES:
        if grad_color is None and grad_material is None:
            if USE_OPENAI_FALLBACK:
                print("[INFO] GradProject unavailable or returned no results. Using OpenAI GPT-4o Vision fallback...")
                grad_color, grad_material, vision_confidence = predict_color_material_openai(
                    item_name, description, item_category, image_urls
                )
                used_fallback = True
                if grad_color or grad_material:
                    grad_data["fallback"] = "openai-gpt4o-vision"
                    grad_data["vision_confidence"] = vision_confidence
            else:
                print("[WARNING] GradProject unavailable or returned no results. OpenAI fallback is disabled (USE_OPENAI_FALLBACK=False).")

    # Step 2: Build OpenAI prompt with grad hints
    input_text = f"""Item Name: {item_name}
Variant Name: {variant_name}
Description: {description}
Vendor Category: {vendor_category}
Shopping Category: {shopping_category}
Shopping Subcategory: {shopping_subcategory}
Item Category: {item_category}"""

    # Add color and material hints from grad model (only for grad-supported categories)
    grad_hints = ""
    color_material_hint = ""
    if grad_color:
        grad_hints += f"\nDetected Color (from image analysis): {grad_color}"
        color_material_hint += "\n- Color: use the detected color from image analysis if provided above"
    if grad_material:
        grad_hints += f"\nDetected Material (from image analysis): {grad_material}"
        color_material_hint += "\n- Material: use the detected material from image analysis if provided above"

    # When no image-based color was detected but a variant_name is present,
    # explicitly instruct the model to extract color/size/material from it.
    if not grad_color and variant_name:
        color_material_hint += f"\n- Color: the Variant Name is \"{variant_name}\" — extract the color from it (e.g. 'Light Beige/S/Polyester' → Color: Light Beige)"
    if not grad_material and variant_name:
        color_material_hint += f"\n- Material and Size: also extract these from the Variant Name if present"

    prompt = f"""
You are a strict AI attribute extractor for e-commerce products.
Analyze the item below and extract ONLY attributes that can be clearly inferred.
Do NOT guess, do NOT add explanations, do NOT include extra text, do NOT use markdown formatting.
Leave unknown attributes empty.

{input_text}{grad_hints}

INSTRUCTIONS:
- Fill only known attributes; leave others empty
- Use concise English values
- If product images are attached, analyze them carefully — they are the strongest signal for Color, Pattern, Material, Fashion Type, Gender, and other visual attributes. Prefer what you see in the images over what the description claims when they disagree.
- Variant Name often encodes size, color, or other variant-specific info (e.g. "Red - XL", "500ml / Blue"). Parse it carefully to extract Size, Color, and any other relevant attributes it contains.
- Gender: choose strictly from ["Women", "Men", "Unisex women, Unisex men", "Girls", "Boys", "Unisex girls, unisex boys"]
- Generic Name: identify the main item (e.g. if "Matelda Chocolate cake 120 grams" → Generic Name: "cake")
- Product Name: the product name without size/quantity (e.g. "Matelda Chocolate cake"){color_material_hint}
- Keep the output clean and structured exactly as below
- DO NOT use markdown code blocks (```)
- DO NOT include "None", "unknown", "N/A" - use empty string instead

OUTPUT FORMAT (exactly, no deviations):
{template}

Output ONLY the above format. NO markdown, NO extra lines or explanations.
"""

    # Pass images to the main extraction call so the model can use vision for
    # color, pattern, material, fashion_type, gender, etc. — not just text hints.
    result = run_openai_model(prompt, images=image_urls if image_urls else None)
    result = merge_with_template(template, result)
    return result, grad_data


def extract_single_item_attributes(item_data, index):
    """Extract AI attributes for a single item - used for parallel processing"""
    try:
        item_name = item_data.get('item_name', '')
        variant_name = item_data.get('variant_name', '')
        description = item_data.get('description', '')
        vendor_category = item_data.get('vendor_category', '')
        shopping_category = item_data.get('shopping_category', '')
        shopping_subcategory = item_data.get('shopping_subcategory', '')
        item_category = item_data.get('item_category', '')
        images = item_data.get('images', [])

        if not all([item_name, shopping_category, item_category]):
            return index, {
                "success": False,
                "message": "item_name, shopping_category, and item_category are required",
                "ai_attributes": "",
                "ai_attributes_array": [],
                "grad_model_used": False,
                "grad_predictions": None
            }

        # Extract attributes
        attributes, grad_data = extract_ai_attributes(
            item_name, description, vendor_category,
            shopping_category, shopping_subcategory, item_category,
            images, variant_name=variant_name
        )

        # Check if there's a warning from grad model
        warning_message = None
        if grad_data and "warning" in grad_data:
            warning_message = grad_data["warning"]

        result = {
            "success": True,
            "message": "AI attributes extracted successfully.",
            "ai_attributes": attributes,
            "ai_attributes_array": attributes.split('\n') if attributes else [],
            "grad_model_used": bool(grad_data) and "warning" not in grad_data,
            "grad_predictions": grad_data if grad_data and "warning" not in grad_data else None,
            "warning": warning_message
        }

        return index, result

    except Exception as e:
        return index, {
            "success": False,
            "message": str(e),
            "ai_attributes": "",
            "ai_attributes_array": [],
            "grad_model_used": False,
            "grad_predictions": None
        }


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
        shopping_category_override = data.get('shopping_category_override', '')

        if not item_name:
            return jsonify({"error": "item_name is required"}), 400

        # Step 1: Shopping Category
        override_clean = shopping_category_override.lower().strip() if shopping_category_override else ''
        if override_clean and override_clean in shoppingCategory:
            shopping_cat = override_clean
        else:
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
        confidence = calculate_confidence(vendor_category, shopping_cat, item_cat, description)

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
        shopping_category_override = data.get('shopping_category_override', '')
        csv_override_clean = shopping_category_override.lower().strip() if shopping_category_override else ''
        if csv_override_clean and csv_override_clean not in shoppingCategory:
            csv_override_clean = ''

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

        def _safe_str(val):
            """Convert pandas cell to string, treating NaN/None as empty."""
            s = str(val).strip() if val is not None else ''
            return '' if s.lower() == 'nan' else s

        # Process each row
        for idx, row in tqdm(df.iterrows(), total=total_rows, desc="Classifying items", unit="row"):
            item_name = _safe_str(row.get('item_name', ''))
            description = _safe_str(row.get('description', ''))
            item_department = _safe_str(row.get('item_department', ''))
            vendor_category = _safe_str(row.get('vendor_category', ''))

            if not item_name:
                print(f"[Row {idx}] Skipped - No item name")
                pd.DataFrame([row]).to_csv(output_path, mode='a', header=False, index=False)
                continue

            stats['rows_with_item_name'] += 1

            # Classify
            if csv_override_clean:
                shopping_cat = csv_override_clean
            else:
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

            confidence = calculate_confidence(vendor_category, shopping_cat, item_cat, description)

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
        shopping_category_override = data.get('shopping_category_override', '')
        override_clean = shopping_category_override.lower().strip() if shopping_category_override else ''
        if override_clean and override_clean not in shoppingCategory:
            override_clean = ''

        if not isinstance(items, list):
            return jsonify({"error": "items must be an array"}), 400

        if len(items) == 0:
            return jsonify({"error": "items array cannot be empty"}), 400

        total_items = len(items)
        print(f"\n[Batch Parallel] Processing {total_items} items with {max_workers} workers...")
        if override_clean:
            print(f"[Batch Parallel] shopping_category_override='{override_clean}' — skipping AI classification for shopping category")

        results = [None] * total_items

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(classify_single_item, item, idx, override_clean or None): idx
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
        variant_name = data.get('variant_name', '')
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

        # Extract attributes
        attributes, grad_data = extract_ai_attributes(
            item_name, description, vendor_category,
            shopping_category, shopping_subcategory, item_category,
            images, variant_name=variant_name
        )

        # Check if there's a warning from grad model
        warning_message = None
        if grad_data and "warning" in grad_data:
            warning_message = grad_data["warning"]

        return jsonify({
            "success": True,
            "message": "AI attributes extracted successfully.",
            "ai_attributes": attributes,
            "ai_attributes_array": attributes.split('\n') if attributes else [],
            "grad_model_used": bool(grad_data) and "warning" not in grad_data,
            "grad_predictions": grad_data if grad_data and "warning" not in grad_data else None,
            "warning": warning_message
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/attributes/extract-batch', methods=['POST'])
def extract_attributes_batch():
    """Process multiple items in parallel for AI attribute extraction"""
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
        print(f"\n[Batch Parallel AI Attributes] Processing {total_items} items with {max_workers} workers...")

        results = [None] * total_items

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(extract_single_item_attributes, item, idx): idx
                for idx, item in enumerate(items)
            }

            for future in as_completed(future_to_index):
                index, result = future.result()
                results[index] = result

        successful = sum(1 for r in results if r.get('success', False))
        failed = total_items - successful

        end_time = time.time()
        elapsed_seconds = end_time - start_time
        hours, remainder = divmod(int(elapsed_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        time_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        print(f"[Batch Parallel AI Attributes] Completed {total_items} items in {time_formatted}")
        print(f"[Batch Parallel AI Attributes] Success: {successful}/{total_items} | Failed: {failed}/{total_items}")

        return jsonify({
            "success": True,
            "results": results,
            "total_items": total_items,
            "successful_extractions": successful,
            "failed_extractions": failed,
            "processing_time": time_formatted,
            "processing_time_seconds": round(elapsed_seconds, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/attributes/test-vision', methods=['POST'])
def test_vision_extraction():
    """
    Test endpoint for OpenAI GPT-4o Vision color/material extraction.
    Directly tests the vision fallback without going through the full pipeline.

    Input JSON:
    {
        "images": ["https://product-image-url.jpg"],
        "item_name": "Product Name",
        "description": "Product description",
        "item_category": "bag"
    }
    """
    try:
        data = request.get_json()

        images = data.get('images', [])
        item_name = data.get('item_name', '')
        description = data.get('description', '')
        item_category = data.get('item_category', '')

        if not images or len(images) == 0:
            return jsonify({
                "success": False,
                "error": "images array is required with at least one image URL"
            }), 400

        if not OPENAI_API_KEY:
            return jsonify({
                "success": False,
                "error": "OPENAI_API_KEY not configured"
            }), 500

        start_time = time.time()

        color, material, confidence_data = predict_color_material_openai(
            item_name, description, item_category, images
        )

        elapsed = time.time() - start_time

        return jsonify({
            "success": True,
            "color": color,
            "material": material,
            "color_confidence": confidence_data.get("color_confidence", 0.0),
            "material_confidence": confidence_data.get("material_confidence", 0.0),
            "model": confidence_data.get("model", "gpt-4o"),
            "images_analyzed": confidence_data.get("images_analyzed", 0),
            "response_time_seconds": round(elapsed, 2)
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/attributes/health', methods=['GET'])
def attributes_health():
    """AI Attributes health check"""
    # Check if external GradProject service is available
    gradproject_available = False
    try:
        response = requests.get(GRADPROJECT_API_URL.replace('/predict', '/health'), timeout=2)
        gradproject_available = response.status_code == 200
    except:
        pass

    return jsonify({
        "status": "healthy",
        "service": "AI Attributes Service",
        "gradproject_service_available": gradproject_available,
        "openai_configured": bool(OPENAI_API_KEY)
    })


# ============================================================
# STANDARDIZATION SERVICE - HELPER FUNCTIONS
# ============================================================

def _attr_key(name):
    """Normalize a display attribute name to a standardization_map key.
    e.g. 'Fashion Type' → 'fashion_type', 'Color' → 'color'
    """
    return name.strip().lower().replace(' ', '_').replace('-', '_')


def standardize_ai_attributes(ai_attributes_input):
    """
    Standardize AI attribute values against the standardization_map.

    Resolution order per attribute:
      1. If value is already a canonical value -> keep it (source: "synonym").
      2. If value matches a known synonym      -> map to canonical (source: "synonym").
      3. Otherwise                             -> call OpenAI to pick the best
                                                  match from the allowed list
                                                  (source: "ai"). This case is
                                                  flagged so the caller knows
                                                  the value was not in the
                                                  synonym list.

    ai_attributes_input: str (newline-separated "Key: Value") or list of such strings.
    Returns (standardized_str, standardized_array, changes_list, ai_fallback_list)
      where ai_fallback_list contains dicts:
        {"attribute": <name>, "original": <raw>, "standardized": <final>}
      for every attribute that had to fall back to the AI model.
    """
    # Normalize input to a list of raw lines
    if isinstance(ai_attributes_input, list):
        raw_lines = ai_attributes_input
    else:
        raw_lines = str(ai_attributes_input).split('\n')

    # Parse into (display_name, value) pairs; value=None means no colon found
    parsed = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        if ':' not in line:
            parsed.append((line, None))
        else:
            name, _, value = line.partition(':')
            parsed.append((name.strip(), value.strip()))

    # Identify which attributes have a standardization mapping and a non-empty value
    to_standardize = []  # (display_name, raw_value, allowed_list)
    for display_name, value in parsed:
        if value is None or value == '':
            continue
        key = _attr_key(display_name)
        if key in standardization_map:
            to_standardize.append((display_name, value, standardization_map[key]))

    # If nothing to standardize, return input unchanged
    if not to_standardize:
        result_lines = [
            f"{n}: {v}" if v is not None else n
            for n, v in parsed
        ]
        return '\n'.join(result_lines), result_lines, [], []

    # ------------------------------------------------------------------
    # Fast path: resolve via local synonyms_map first to save AI tokens.
    # Anything not matched here falls through to the AI model below.
    # ------------------------------------------------------------------
    synonym_resolved = {}  # display_name.lower() -> canonical value
    remaining = []         # entries still needing the AI model

    for display_name, original_value, allowed in to_standardize:
        attr_key = _attr_key(display_name)
        allowed_lower = {v.lower(): v for v in allowed}
        raw_lower = original_value.strip().lower()

        # (a) Already a canonical value (case-insensitive match in allowed list)
        if raw_lower in allowed_lower:
            synonym_resolved[display_name.lower()] = allowed_lower[raw_lower]
            continue

        # (b) Known synonym for this attribute (uses pre-built reverse lookup)
        attr_syns = _SYNONYM_LOOKUP.get(attr_key, {})
        canonical = attr_syns.get(raw_lower)
        if canonical and canonical.lower() in allowed_lower:
            # enforce exact casing from the standardization map
            synonym_resolved[display_name.lower()] = allowed_lower[canonical.lower()]
            continue

        # (c) Needs AI
        remaining.append((display_name, original_value, allowed))

    # If everything was resolved via synonyms, skip the AI call entirely.
    if not remaining:
        standardized_lookup = synonym_resolved
        changes = []
        for display_name, original_value, _allowed in to_standardize:
            final_value = standardized_lookup[display_name.lower()]
            if final_value.lower() != original_value.lower():
                changes.append({
                    "attribute": display_name,
                    "original": original_value,
                    "standardized": final_value,
                    "source": "synonym"
                })

        result_lines = []
        for display_name, value in parsed:
            if value is None:
                result_lines.append(display_name)
            elif value == '':
                result_lines.append(f"{display_name}:")
            else:
                final = standardized_lookup.get(display_name.lower(), value)
                result_lines.append(f"{display_name}: {final}")
        # All resolved via synonyms -> no AI fallback
        return '\n'.join(result_lines), result_lines, changes, []

    # Build a single prompt for ONLY the remaining (unresolved) attributes
    attr_sections = '\n'.join(
        f'- {name}: "{value}"\n  Allowed values: {allowed}'
        for name, value, allowed in remaining
    )

    prompt = f"""You are a strict attribute standardization bot for e-commerce products.
For each attribute below, map the raw value to the BEST MATCH from its allowed list.
DO NOT explain. Return exactly one line per attribute in the format: AttributeName: StandardizedValue

RULES:
- Choose the closest semantic match from the allowed list.
- If the value already matches one entry (case-insensitive), return it in the exact casing shown in the allowed list.
- If no reasonable match exists, return the original value unchanged.
- NEVER return a value that is not in the allowed list (unless using the original fallback).

Attributes to standardize:
{attr_sections}

Output (one line per attribute, same order as above):"""

    raw_output = run_openai_model(prompt)

    # Parse model output into a name→value dict (case-insensitive keys)
    model_values = {}
    for line in raw_output.split('\n'):
        line = line.strip()
        if ':' in line:
            n, _, v = line.partition(':')
            model_values[n.strip().lower()] = v.strip()

    # Validate each standardized value against its allowed list
    standardized_lookup = {}  # display_name.lower() → final value
    changes = []
    ai_fallback = []  # attributes that were NOT in synonyms and required the AI model

    for display_name, original_value, allowed in to_standardize:
        allowed_lower = {v.lower(): v for v in allowed}

        # Prefer synonym-resolved value if present (no AI was called for it)
        if display_name.lower() in synonym_resolved:
            final_value = synonym_resolved[display_name.lower()]
            source = "synonym"
        else:
            raw_std = model_values.get(display_name.lower(), original_value)
            if raw_std.lower() in allowed_lower:
                final_value = allowed_lower[raw_std.lower()]  # enforce exact casing from mapping
            else:
                final_value = original_value  # fallback: keep original
            source = "ai"
            ai_fallback.append({
                "attribute": display_name,
                "original": original_value,
                "standardized": final_value
            })

        standardized_lookup[display_name.lower()] = final_value

        if final_value.lower() != original_value.lower():
            changes.append({
                "attribute": display_name,
                "original": original_value,
                "standardized": final_value,
                "source": source
            })

    # Rebuild output lines in original order
    result_lines = []
    for display_name, value in parsed:
        if value is None:
            result_lines.append(display_name)
        elif value == '':
            result_lines.append(f"{display_name}:")
        else:
            final = standardized_lookup.get(display_name.lower(), value)
            result_lines.append(f"{display_name}: {final}")

    result_str = '\n'.join(result_lines)
    return result_str, result_lines, changes, ai_fallback


def _standardize_single(item_data, index):
    """Standardize attributes for one item — used for parallel batch processing."""
    try:
        ai_attributes = item_data.get('ai_attributes', '')
        ai_attributes_array = item_data.get('ai_attributes_array', [])

        input_data = ai_attributes_array if ai_attributes_array else ai_attributes
        if not input_data:
            return index, {
                "success": False,
                "message": "ai_attributes or ai_attributes_array is required",
                "standardized_attributes": "",
                "standardized_attributes_array": [],
                "changes": []
            }

        std_str, std_array, changes, ai_fallback = standardize_ai_attributes(input_data)
        ai_used = len(ai_fallback) > 0
        if ai_used:
            flagged = ", ".join(f"{f['attribute']} (\"{f['original']}\")" for f in ai_fallback)
            message = (
                f"AI fallback used for {len(ai_fallback)} attribute(s) not found in synonyms: {flagged}. "
                f"Consider adding them to standardization_synonyms.py to save tokens."
            )
        else:
            message = "All attributes resolved via synonyms map (no AI call needed)."
        return index, {
            "success": True,
            "message": message,
            "standardized_attributes": std_str,
            "standardized_attributes_array": std_array,
            "changes": changes,
            "changes_count": len(changes),
            "ai_fallback_used": ai_used,
            "ai_fallback_attributes": ai_fallback
        }
    except Exception as e:
        return index, {
            "success": False,
            "message": str(e),
            "standardized_attributes": "",
            "standardized_attributes_array": [],
            "changes": []
        }


# ============================================================
# STANDARDIZATION SERVICE - ROUTES
# ============================================================

@app.route('/api/attributes/standardize', methods=['POST'])
def standardize_attributes():
    """
    Standardize AI attribute values using the standardization mapping.

    Input JSON:
    {
        "ai_attributes": "Color: Navy Blue\\nSize: X-Large\\nGender: Female",
        // OR
        "ai_attributes_array": ["Color: Navy Blue", "Size: X-Large", "Gender: Female"]
    }

    Returns:
    {
        "success": true,
        "standardized_attributes": "Color: Dark Blue\\nSize: XL\\nGender: Women",
        "standardized_attributes_array": ["Color: Dark Blue", "Size: XL", "Gender: Women"],
        "changes": [
            {"attribute": "Color", "original": "Navy Blue", "standardized": "Dark Blue"},
            {"attribute": "Size", "original": "X-Large", "standardized": "XL"},
            {"attribute": "Gender", "original": "Female", "standardized": "Women"}
        ],
        "changes_count": 3
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        ai_attributes = data.get('ai_attributes', '')
        ai_attributes_array = data.get('ai_attributes_array', [])

        input_data = ai_attributes_array if ai_attributes_array else ai_attributes
        if not input_data:
            return jsonify({"error": "ai_attributes or ai_attributes_array is required"}), 400

        if not OPENAI_API_KEY:
            return jsonify({"error": "OpenAI API key not configured"}), 500

        std_str, std_array, changes, ai_fallback = standardize_ai_attributes(input_data)
        ai_used = len(ai_fallback) > 0
        if ai_used:
            flagged = ", ".join(f"{f['attribute']} (\"{f['original']}\")" for f in ai_fallback)
            message = (
                f"AI fallback used for {len(ai_fallback)} attribute(s) not found in synonyms: {flagged}. "
                f"Consider adding them to standardization_synonyms.py to save tokens."
            )
        else:
            message = "All attributes resolved via synonyms map (no AI call needed)."

        return jsonify({
            "success": True,
            "message": message,
            "standardized_attributes": std_str,
            "standardized_attributes_array": std_array,
            "changes": changes,
            "changes_count": len(changes),
            "ai_fallback_used": ai_used,
            "ai_fallback_attributes": ai_fallback
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/attributes/standardize-batch', methods=['POST'])
def standardize_attributes_batch():
    """
    Standardize AI attributes for multiple items in parallel.

    Input JSON:
    {
        "items": [
            {"ai_attributes": "Color: Navy Blue\\nSize: X-Large"},
            {"ai_attributes_array": ["Color: Red", "Size: Small"]}
        ],
        "max_workers": 3
    }
    """
    try:
        start_time = time.time()
        data = request.get_json()

        if not data or 'items' not in data:
            return jsonify({"error": "items array is required"}), 400

        items = data.get('items', [])
        max_workers = data.get('max_workers', 3)

        if not isinstance(items, list) or len(items) == 0:
            return jsonify({"error": "items must be a non-empty array"}), 400

        if not OPENAI_API_KEY:
            return jsonify({"error": "OpenAI API key not configured"}), 500

        total_items = len(items)
        print(f"\n[Batch Standardization] Processing {total_items} items with {max_workers} workers...")

        results = [None] * total_items

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(_standardize_single, item, idx): idx
                for idx, item in enumerate(items)
            }
            for future in as_completed(future_to_index):
                index, result = future.result()
                results[index] = result

        successful = sum(1 for r in results if r.get('success', False))
        failed = total_items - successful

        elapsed = time.time() - start_time
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        time_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        print(f"[Batch Standardization] Completed {total_items} items in {time_formatted}")
        print(f"[Batch Standardization] Success: {successful}/{total_items} | Failed: {failed}/{total_items}")

        return jsonify({
            "success": True,
            "results": results,
            "total_items": total_items,
            "successful_standardizations": successful,
            "failed_standardizations": failed,
            "processing_time": time_formatted,
            "processing_time_seconds": round(elapsed, 2)
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# DESCRIPTION GENERATION SERVICE - HELPER FUNCTIONS
# ============================================================

def generate_item_description(item_name, item_department, variant_name,
                               ai_attributes=None):
    """
    Generate a description for an item using OpenAI.

    Primary inputs: item_name, item_department, variant_name
    Optional: ai_attributes dict with color, material, style, etc.
    """

    # Build hints from optional AI attributes
    hints = ""
    if ai_attributes:
        if ai_attributes.get('color'):
            hints += f"\n- Color: {ai_attributes['color']}"
        if ai_attributes.get('material'):
            hints += f"\n- Material: {ai_attributes['material']}"
        if ai_attributes.get('style'):
            hints += f"\n- Style: {ai_attributes['style']}"
        if ai_attributes.get('pattern'):
            hints += f"\n- Pattern: {ai_attributes['pattern']}"
        if ai_attributes.get('size'):
            hints += f"\n- Size: {ai_attributes['size']}"
        # Allow any additional attributes
        for key, value in ai_attributes.items():
            if key not in ['color', 'material', 'style', 'pattern', 'size'] and value:
                hints += f"\n- {key.replace('_', ' ').title()}: {value}"

    prompt = f"""You are an e-commerce product description writer. Generate a concise, professional product description.

PRIMARY INFORMATION:
- Item Name: {item_name}
- Department: {item_department if item_department else 'Not specified'}
- Variant: {variant_name if variant_name else 'Not specified'}
{f"PRODUCT ATTRIBUTES (incorporate naturally):{hints}" if hints else ""}

INSTRUCTIONS:
1. Write a clear, engaging product description (2-4 sentences)
2. Focus on key features and benefits
3. If attributes are provided, incorporate them naturally
4. Use professional e-commerce language
5. Do NOT include prices or availability
6. Do NOT make up specific measurements or specifications
7. Keep it concise and informative

OUTPUT: Write ONLY the product description, no additional text or formatting.
"""

    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": OPENAI_MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a professional e-commerce copywriter who creates concise, engaging product descriptions."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 200,
            "temperature": 0.7
        }

        print(f"[DEBUG] Calling OpenAI API for description generation...")
        r = requests.post(OPENAI_API_URL, json=payload, headers=headers)
        r.raise_for_status()
        description = r.json()["choices"][0]["message"]["content"].strip()

        # Clean up any quotes or extra formatting
        description = description.strip('"\'')

        return description

    except Exception as e:
        print(f"[ERROR] Description generation failed: {str(e)}")
        raise


# ============================================================
# DESCRIPTION GENERATION SERVICE - ROUTES
# ============================================================

@app.route('/api/description/generate', methods=['POST'])
def generate_description():
    """
    Generate a product description for items without descriptions.

    Input JSON:
    {
        "item_name": "Blue Cotton T-Shirt",           # Required
        "item_department": "Men's Clothing",          # Optional
        "variant_name": "Large",                       # Optional
        "ai_attributes": {                             # Optional - hints for description
            "color": "blue",
            "material": "cotton",
            "style": "casual",
            "pattern": "solid",
            ...any other attributes
        }
    }

    Returns:
    {
        "success": true,
        "description": "Generated product description..."
    }
    """
    try:
        data = request.get_json()

        # Required field
        item_name = data.get('item_name', '').strip()
        if not item_name:
            return jsonify({
                "success": False,
                "error": "item_name is required"
            }), 400

        # Optional fields
        item_department = data.get('item_department', '').strip()
        variant_name = data.get('variant_name', '').strip()
        ai_attributes = data.get('ai_attributes', {})

        # Check OpenAI configuration
        if not OPENAI_API_KEY:
            return jsonify({
                "success": False,
                "error": "OpenAI API key not configured"
            }), 500

        # Generate description
        description = generate_item_description(
            item_name=item_name,
            item_department=item_department,
            variant_name=variant_name,
            ai_attributes=ai_attributes
        )

        return jsonify({
            "success": True,
            "description": description
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/description/generate-batch', methods=['POST'])
def generate_description_batch():
    """
    Generate descriptions for multiple items in parallel.

    Input JSON:
    {
        "items": [
            {
                "item_name": "...",
                "item_department": "...",
                "variant_name": "...",
                "ai_attributes": {
                    "color": "...",
                    "material": "...",
                    ...
                }
            }
        ],
        "max_workers": 3  # Optional, default 3
    }
    """
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
        print(f"\n[Batch Description Generation] Processing {total_items} items with {max_workers} workers...")

        def generate_single_description(item_data, index):
            """Process a single item for description generation"""
            try:
                item_name = item_data.get('item_name', '').strip()
                if not item_name:
                    return index, {
                        "success": False,
                        "error": "item_name is required",
                        "description": ""
                    }

                description = generate_item_description(
                    item_name=item_name,
                    item_department=item_data.get('item_department', '').strip(),
                    variant_name=item_data.get('variant_name', '').strip(),
                    ai_attributes=item_data.get('ai_attributes', {})
                )

                return index, {
                    "success": True,
                    "description": description
                }

            except Exception as e:
                return index, {
                    "success": False,
                    "error": str(e),
                    "description": ""
                }

        results = [None] * total_items

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(generate_single_description, item, idx): idx
                for idx, item in enumerate(items)
            }

            for future in as_completed(future_to_index):
                index, result = future.result()
                results[index] = result

        successful = sum(1 for r in results if r.get('success', False))
        failed = total_items - successful

        end_time = time.time()
        elapsed_seconds = end_time - start_time
        hours, remainder = divmod(int(elapsed_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        time_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        print(f"[Batch Description Generation] Completed {total_items} items in {time_formatted}")
        print(f"[Batch Description Generation] Success: {successful}/{total_items} | Failed: {failed}/{total_items}")

        return jsonify({
            "success": True,
            "results": results,
            "total_items": total_items,
            "successful_generations": successful,
            "failed_generations": failed,
            "processing_time": time_formatted,
            "processing_time_seconds": round(elapsed_seconds, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# GRAMMAR CHECK SERVICE - ROUTES
# ============================================================

@app.route('/api/grammar/check', methods=['POST'])
def grammar_check():
    """
    
    Grammar check endpoint - supports both single and batch processing
    Uses LanguageTool locally (no external service required)

    Input JSON (single):
    { "text": "This are wrong" }

    Input JSON (batch/array):
    [{ "text": "This are wrong" }, { "text": "I has error" }]

    Returns corrected text with original text and changes made
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Get grammar tool instance
        tool = get_grammar_tool()

        # Store original text(s)
        is_batch = isinstance(data, list)

        if is_batch:
            # Batch processing
            results = []
            for item in data:
                original_text = item.get("text", "")
                if not original_text:
                    results.append({
                        "original_text": "",
                        "corrected_text": "",
                        "has_changes": False,
                        "changes_summary": "No text provided"
                    })
                    continue

                # Check and correct grammar
                matches = tool.check(original_text)
                corrected_text = language_tool_python.utils.correct(original_text, matches)

                results.append({
                    "original_text": original_text,
                    "corrected_text": corrected_text,
                    "has_changes": original_text != corrected_text,
                    "changes_summary": f"Changed from '{original_text}' to '{corrected_text}'" if original_text != corrected_text else "No changes",
                    "errors_found": len(matches)
                })

            return jsonify(results), 200
        else:
            # Single processing
            original_text = data.get("text", "")

            if not original_text:
                return jsonify({
                    "original_text": "",
                    "corrected_text": "",
                    "has_changes": False,
                    "changes_summary": "No text provided",
                    "errors_found": 0
                }), 200

            # Check and correct grammar
            matches = tool.check(original_text)
            corrected_text = language_tool_python.utils.correct(original_text, matches)

            return jsonify({
                "original_text": original_text,
                "corrected_text": corrected_text,
                "has_changes": original_text != corrected_text,
                "changes_summary": f"Changed from '{original_text}' to '{corrected_text}'" if original_text != corrected_text else "No changes",
                "errors_found": len(matches)
            }), 200

    except Exception as e:
        return jsonify({
            "error": f"Grammar check failed: {str(e)}",
            "message": "Error processing grammar check"
        }), 500


# ============================================================
# TRANSLATION SERVICE - ROUTES
# ============================================================

@app.route('/api/translation/translate', methods=['POST'])
def translate_endpoint():
    """
    Translate English to Arabic (single or batch)

    Input JSON (single):
    { "text": "Your text here", "vendor_name": "VendorX" }

    Input JSON (batch):
    { "texts": ["Text 1", "Text 2"], "vendor_names": ["Vendor1", "Vendor2"] }
    """
    try:
        data = request.get_json(force=True)
        vendor_name = data.get("vendor_name", "")

        if "texts" in data:
            if not isinstance(data["texts"], list):
                return jsonify({"error": "'texts' must be a list"}), 400

            vendor_names = data.get("vendor_names", [])
            translations = translate_batch(data["texts"])

            return jsonify({
                "translations": translations,
                "vendor_names": vendor_names if vendor_names else [vendor_name] * len(translations)
            })

        if "text" in data:
            # Support both single string and array in "text" field
            if isinstance(data["text"], list):
                # Treat as batch translation
                vendor_names = data.get("vendor_names", [])
                translations = translate_batch(data["text"])

                return jsonify({
                    "translations": translations,
                    "vendor_names": vendor_names if vendor_names else [vendor_name] * len(translations)
                })
            else:
                # Single translation
                return jsonify({
                    "translation": translate_to_arabic(data["text"]),
                    "vendor_name": vendor_name
                })

        return jsonify({"error": "Provide 'text' (string or array) or 'texts' (array)"}), 400

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

        confidence = calculate_confidence(vendor_category, shopping_cat, item_cat, description)
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

        attributes, grad_data = extract_ai_attributes(
            item_name, description, vendor_category,
            shopping_cat, shopping_subcat, item_cat,
            images
        )

        ai_attributes_result["english"] = attributes
        ai_attributes_result["grad_model_used"] = bool(grad_data) and "warning" not in grad_data
        ai_attributes_result["grad_predictions"] = grad_data if grad_data and "warning" not in grad_data else None

        # Add warning if GradProject unavailable
        if grad_data and "warning" in grad_data:
            ai_attributes_result["warning"] = grad_data["warning"]

        # Step 3: Translation (if requested)
        if should_translate and attributes:
            arabic_translation = translate_to_arabic(attributes)
            ai_attributes_result["arabic"] = arabic_translation

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
# KEYWORDS SERVICE - HELPER FUNCTIONS
# ============================================================

COMMON_COLORS = {
    "black", "white", "red", "blue", "green", "yellow", "orange", "purple",
    "pink", "brown", "grey", "gray", "silver", "gold", "navy", "beige",
    "ivory", "cream", "maroon", "violet", "teal", "cyan", "magenta", "coral",
    "turquoise", "khaki", "olive", "indigo", "lavender", "rose", "mint",
    "charcoal", "tan", "burgundy", "aqua", "lime", "fuchsia", "peach",
    "multicolor", "multi-color", "multicolour", "multi-colour"
}


def phrase_contains_color(phrase):
    """Return True if the phrase contains any color word."""
    words = re.findall(r'[a-z]+', phrase.lower())
    return bool(COMMON_COLORS.intersection(words))


def refine_search_keywords(skw_list, dsw_list):
    """
    Use OpenAI to generate/refine DSW from SKW following the rules:
    - No SKW phrases repeated in DSW
    - Colors only in DSW, never in SKW (colored SKW phrases are moved to DSW)
    Returns (clean_skw_list, refined_dsw_list)
    """
    skw_text = ", ".join(skw_list)
    dsw_text = ", ".join(dsw_list) if dsw_list else ""

    prompt = f"""You are helping generate product search keywords for an e-commerce platform.

You are given:
SKW (Search Keywords): {skw_text}
DSW (Derived Search Words): {dsw_text}

Your task: Generate a refined set of SKW and DSW following these STRICT rules:

RULES:
1. SKW must NOT contain any color words (black, red, blue, green, white, yellow, orange, purple, pink, brown, grey, gray, silver, gold, navy, beige, ivory, cream, maroon, violet, teal, cyan, magenta, coral, turquoise, khaki, olive, indigo, lavender, rose, mint, charcoal, tan, burgundy, aqua, lime, fuchsia, peach, multicolor, etc.)
   - Any current SKW phrase that contains a color must be MOVED to DSW, not kept in SKW
2. DSW must NOT repeat any phrase that is already in SKW (exact match, case-insensitive)
3. Generate additional useful DSW phrases based on the SKW — include color-based variations in DSW
4. DSW phrases should be related search variations, synonyms, or combinations
5. Do not duplicate entries within SKW or within DSW

OUTPUT FORMAT (strictly follow — no extra text, no explanations):
SKW: <comma-separated list>
DSW: <comma-separated list>
"""

    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": OPENAI_MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a strict search keyword refinement assistant. Follow instructions exactly."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.3
        }

        r = requests.post(OPENAI_API_URL, json=payload, headers=headers)
        r.raise_for_status()
        result = r.json()["choices"][0]["message"]["content"].strip()

        refined_skw = []
        refined_dsw = []

        for line in result.split('\n'):
            line = line.strip()
            if line.lower().startswith('skw:'):
                raw = line.split(':', 1)[1].strip()
                refined_skw = [p.strip() for p in raw.split(',') if p.strip()]
            elif line.lower().startswith('dsw:'):
                raw = line.split(':', 1)[1].strip()
                refined_dsw = [p.strip() for p in raw.split(',') if p.strip()]

        # Post-process: enforce color rule locally as a safety net
        skw_set_lower = {p.lower() for p in refined_skw}
        final_skw = []
        overflow_to_dsw = []
        for phrase in refined_skw:
            if phrase_contains_color(phrase):
                overflow_to_dsw.append(phrase)
            else:
                final_skw.append(phrase)

        # Merge overflow into DSW, avoid duplicating existing DSW entries
        dsw_lower = {p.lower() for p in refined_dsw}
        for phrase in overflow_to_dsw:
            if phrase.lower() not in dsw_lower:
                refined_dsw.append(phrase)
                dsw_lower.add(phrase.lower())

        # Remove DSW entries that duplicate SKW
        skw_final_lower = {p.lower() for p in final_skw}
        final_dsw = [p for p in refined_dsw if p.lower() not in skw_final_lower]

        return final_skw, final_dsw

    except Exception as e:
        print(f"[ERROR] Keyword refinement failed: {str(e)}")
        raise


# ============================================================
# KEYWORDS SERVICE - ROUTES
# ============================================================

@app.route('/api/keywords/refine', methods=['POST'])
def refine_keywords():
    """
    Refine product search keywords (SKW / DSW).

    Input JSON:
    {
        "skw": "notebook, spiral hardcover notebook, black hard cover notebook",
        "dsw": "hard cover spiral notebook, spiral notebook"   // optional
    }

    SKW and DSW can be either a comma-separated string or a list of strings.

    Rules enforced:
    - Colors must not appear in SKW (colored phrases are moved to DSW)
    - DSW must not repeat any phrase already in SKW
    - Additional relevant DSW phrases are generated via AI

    Returns:
    {
        "success": true,
        "skw": ["notebook", "spiral hardcover notebook", ...],
        "dsw": ["black hard cover notebook", "spiral bound notebook", ...],
        "skw_string": "notebook, spiral hardcover notebook, ...",
        "dsw_string": "black hard cover notebook, spiral bound notebook, ..."
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        raw_skw = data.get('skw', '')
        raw_dsw = data.get('dsw', '')

        # Normalise input — accept both string and list
        if isinstance(raw_skw, list):
            skw_list = [p.strip() for p in raw_skw if str(p).strip()]
        else:
            skw_list = [p.strip() for p in str(raw_skw).split(',') if p.strip()]

        if isinstance(raw_dsw, list):
            dsw_list = [p.strip() for p in raw_dsw if str(p).strip()]
        else:
            dsw_list = [p.strip() for p in str(raw_dsw).split(',') if p.strip()]

        if not skw_list:
            return jsonify({"success": False, "error": "skw is required and cannot be empty"}), 400

        if not OPENAI_API_KEY:
            return jsonify({"success": False, "error": "OpenAI API key not configured"}), 500

        print(f"[Keywords] Refining {len(skw_list)} SKW and {len(dsw_list)} DSW phrases...")
        refined_skw, refined_dsw = refine_search_keywords(skw_list, dsw_list)

        return jsonify({
            "success": True,
            "skw": refined_skw,
            "dsw": refined_dsw,
            "skw_string": ", ".join(refined_skw),
            "dsw_string": ", ".join(refined_dsw)
        }), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# GLOBAL HEALTH ENDPOINT
# ============================================================

@app.route('/health', methods=['GET'])
def global_health():
    """Global health check for all services"""
    # Check if external GradProject service is available
    gradproject_available = False
    try:
        response = requests.get(GRADPROJECT_API_URL.replace('/predict', '/health'), timeout=2)
        gradproject_available = response.status_code == 200
    except:
        pass

    return jsonify({
        "status": "healthy",
        "services": ["category", "ai_attributes", "standardization", "description", "grammar", "translation", "pipeline", "keywords"],
        "models": {
            "openai": bool(OPENAI_API_KEY),
            "gradproject_service": gradproject_available,
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
    print(f"  GradProject Service: {GRADPROJECT_API_URL}")

    print("\nEndpoints:")
    print("  Category Service:")
    print("    POST /api/category/classify         - Single item classification")
    print("    POST /api/category/classify-csv     - Process CSV file")
    print("    POST /api/category/classify-batch   - Parallel batch classification")

    print("\n  AI Attributes Service:")
    print("    POST /api/attributes/extract        - Extract AI attributes")
    print("    POST /api/attributes/extract-batch  - Parallel batch AI attribute extraction")
    print("    POST /api/attributes/test-vision    - Test GPT-4o vision color/material extraction")
    print("    POST /api/attributes/standardize    - Standardize AI attribute values")
    print("    POST /api/attributes/standardize-batch - Parallel batch standardization")

    print("\n  Description Generation Service:")
    print("    POST /api/description/generate      - Generate item description")
    print("    POST /api/description/generate-batch - Batch description generation")

    print("\n  Grammar Check Service:")
    print("    POST /api/grammar/check             - Grammar check (single or batch)")

    print("\n  Translation Service:")
    print("    POST /api/translation/translate     - English -> Arabic translation")

    print("\n  Pipeline Service:")
    print("    POST /api/pipeline/process          - End-to-end processing")
    print("    POST /api/pipeline/process-batch    - Batch pipeline processing")

    print("\n  Keywords Service:")
    print("    POST /api/keywords/refine           - Refine SKW / DSW search keywords")

    print("\n  Global:")
    print("    GET  /health                        - Global health check")

    print("\n" + "="*70)
    print(f"Starting unified server on http://{FLASK_HOST}:{FLASK_PORT}")
    print("="*70 + "\n")

    # Note: GradProject is now an external service
    print("[INFO] GradProject runs as external service - ensure it's running on port 5000")
    print("")

    app.run(debug=FLASK_DEBUG, host=FLASK_HOST, port=FLASK_PORT)
