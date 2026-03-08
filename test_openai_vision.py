"""
Test script for OpenAI GPT-4o Vision - Color & Material extraction from product images.
Run standalone to test vision capabilities, or use the Flask endpoint for Postman testing.

Usage:
    Standalone:  python test_openai_vision.py
    Postman:     POST http://localhost:6002/api/attributes/test-vision
"""

import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
VISION_MODEL = "gpt-4o"


def extract_color_material_from_image(image_urls, item_name="", description="", item_category=""):
    """
    Use GPT-4o vision to extract color and material from product images.

    Args:
        image_urls: list of image URLs to analyze
        item_name: product name (optional context)
        description: product description (optional context)
        item_category: product category (optional context)

    Returns:
        dict with color, material, and confidence scores
    """
    if not OPENAI_API_KEY:
        return {"error": "OPENAI_API_KEY not set in .env"}

    prompt_text = f"""You are a product attribute extractor. Analyze the product image(s) and identify the color and material.

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

    # Build content with images
    user_content = []
    for url in image_urls[:3]:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": url, "detail": "low"}
        })
    user_content.append({"type": "text", "text": prompt_text})

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise product attribute extractor. Analyze the images carefully to determine color and material."
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        "max_tokens": 100,
        "temperature": 0.1
    }

    start_time = time.time()
    r = requests.post(OPENAI_API_URL, json=payload, headers=headers)
    elapsed = time.time() - start_time
    r.raise_for_status()

    result = r.json()["choices"][0]["message"]["content"].strip()

    # Parse response
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
                pass
        elif line.startswith('color:'):
            val = line.split(':', 1)[1].strip()
            if val and val not in ['none', 'unknown', 'n/a', 'empty', '']:
                color = val
        elif line.startswith('material_confidence:'):
            val = line.split(':', 1)[1].strip()
            try:
                material_confidence = float(val)
            except ValueError:
                pass
        elif line.startswith('material:'):
            val = line.split(':', 1)[1].strip()
            if val and val not in ['none', 'unknown', 'n/a', 'empty', '']:
                material = val

    return {
        "color": color,
        "color_confidence": color_confidence,
        "material": material,
        "material_confidence": material_confidence,
        "model": VISION_MODEL,
        "images_analyzed": len(image_urls[:3]),
        "response_time_seconds": round(elapsed, 2),
        "raw_response": result
    }


# ============================================================
# STANDALONE TEST
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("OpenAI GPT-4o Vision - Color & Material Extraction Test")
    print("=" * 60)

    if not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY not found in .env file")
        exit(1)

    print(f"[OK] API Key: {OPENAI_API_KEY[:10]}...")
    print(f"[OK] Model: {VISION_MODEL}")
    print()

    # Test cases - replace with your own product image URLs
    test_cases = [
        {
            "name": "Test 1 - Red Leather Bag",
            "images": [
                "https://images.unsplash.com/photo-1548036328-c9fa89d128fa?w=400"
            ],
            "item_name": "Classic Leather Handbag",
            "description": "A stylish handbag for everyday use",
            "item_category": "bag"
        },
        {
            "name": "Test 2 - Blue Denim Jacket",
            "images": [
                "https://images.unsplash.com/photo-1576995853123-5a10305d93c0?w=400"
            ],
            "item_name": "Denim Jacket",
            "description": "Classic denim jacket with button closure",
            "item_category": "jacket"
        },
        {
            "name": "Test 3 - Wooden Table",
            "images": [
                "https://images.unsplash.com/photo-1533090161767-e6ffed986c88?w=400"
            ],
            "item_name": "Dining Table",
            "description": "Modern dining table",
            "item_category": "table"
        },
    ]

    for i, test in enumerate(test_cases):
        print(f"\n{'─' * 50}")
        print(f"  {test['name']}")
        print(f"  Item: {test['item_name']}")
        print(f"  Images: {len(test['images'])}")
        print(f"{'─' * 50}")

        try:
            result = extract_color_material_from_image(
                image_urls=test["images"],
                item_name=test["item_name"],
                description=test["description"],
                item_category=test["item_category"]
            )

            print(f"  Color:      {result['color']} (confidence: {result['color_confidence']})")
            print(f"  Material:   {result['material']} (confidence: {result['material_confidence']})")
            print(f"  Time:       {result['response_time_seconds']}s")
            print(f"  Raw:        {result['raw_response']}")

        except Exception as e:
            print(f"  [ERROR] {str(e)}")

    # ============================================================
    # POSTMAN TEST INSTRUCTIONS
    # ============================================================
    print("\n\n" + "=" * 60)
    print("POSTMAN TESTING INSTRUCTIONS")
    print("=" * 60)
    print("""
1. Start the Flask server:
   python app.py

2. Test via the existing attributes endpoint (uses GPT-4o vision as fallback):
   POST http://localhost:6002/api/attributes/extract
   Content-Type: application/json

   {
       "item_name": "Classic Leather Handbag",
       "description": "A stylish leather handbag",
       "vendor_category": "bags",
       "shopping_category": "fashion",
       "shopping_subcategory": "women accessories",
       "item_category": "bag",
       "images": [
           "https://your-product-image-url.jpg"
       ]
   }

3. Test the dedicated vision test endpoint:
   POST http://localhost:6002/api/attributes/test-vision
   Content-Type: application/json

   {
       "images": [
           "https://your-product-image-url.jpg"
       ],
       "item_name": "Classic Leather Handbag",
       "description": "A stylish leather handbag",
       "item_category": "bag"
   }

4. Expected response:
   {
       "success": true,
       "color": "brown",
       "color_confidence": 0.95,
       "material": "leather",
       "material_confidence": 0.9,
       "model": "gpt-4o",
       "images_analyzed": 1,
       "response_time_seconds": 2.3
   }
""")
