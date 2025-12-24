using hugging face "BLIP-MAE"

in flask api

preprocess images URLS
and get majoroty voting

POST /extract 

using 
images (optianal)
description
category
attributes


Downloads images from URLs and saves them as temporary files.

Moves the model to GPU if available, otherwise uses CPU.

Measures time for each stage: image download, model inference, and total request processing.

Runs the model on all images and aggregates predictions across multiple images to select the highest-confidence values per attribute.

Converts NumPy types in the results to native Python types for JSON serialization.

Cleans up temporary image files after processing.

Returns a JSON response including aggregated predictions, metadata (number of images processed, time taken), and original description/category.

# ------------------------------------------------------------------------------------------------------------------------------------------------ #

"fashion": ["sleeve", "color", "type", "pattern", "material", "style", "neck", "gender", "brand"]

 "home_garden": ["purpose", "measurements", "color", "number of pieces", "generic name", "material", "brand", "weight", "features", "components",
    "size", "product name", "shape", "type of decor style", "pattern", "scent", "number of lights", "country of origin", "light type",
    "occasion", "finishing", "service count", "light fixture form", "watts"]