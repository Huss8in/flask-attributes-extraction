# ------------------------------------------------------------------------------------------------------------------------------ #
# Common synonyms for standardization.
#
# Purpose: avoid calling the AI model for obvious mappings (saves tokens).
# Structure: { attribute_key: { canonical_value: [synonym, synonym, ...] } }
#
# - attribute_key matches the keys in standardization_map (e.g. "color", "size", "gender").
# - canonical_value MUST exist in the corresponding standardization_map list,
#   otherwise the synonyms are ignored at validation time.
# - Synonyms are matched case-insensitively after stripping whitespace.
# - The canonical value itself is auto-added as a synonym at import time,
#   so you don't need to repeat it in the list.
#
# Extend freely as you spot recurring raw values in production.
# ------------------------------------------------------------------------------------------------------------------------------ #

synonyms_map = {
# ========================================================================
    "color": {
        # --- Yellows ---
        "Yellow":        ["duck egg", "ochre", "neon yellow", "yellow green", "mango yellow", "sunflower", "amber"],
        "Light Yellow":  ["cream", "sun yellow", "raffia", "light canary", "wheat", "honey"],
        "Dark Yellow":   ["dijon", "mustard yellow"],

        # --- Oranges ---
        "Orange":        ["peach", "dusty orange", "nasturtium", "teracotta", "terracotta orange",
                          "mandarin", "mango", "coral", "stella"],
        "Light Orange":  ["bright orange", "light peach", "pale coral", "bright peach", "simon",
                          "somon", "melon", "apricot", "electric orange"],
        "Dark Orange":   ["burnt orange", "cardinal", "carrot"],

        # --- Reds ---
        "Red":           ["mahogany", "rosso", "ruby", "rose red", "brick", "firebrick",
                          "raspberry", "brick red", "hot red", "lollipop", "crimson", "rust"],
        "Light Red":     ["watermelon red"],
        "Dark Red":      ["sienna", "maroon", "jester red", "burgundy", "wine", "cranberry",
                          "bordeaux", "wine red", "winered", "rumba red", "cherry"],

        # --- Pinks ---
        "Pink":          ["barbie pink", "hot pink", "rose", "dusty pink", "baby pink",
                          "rose gold", "cedar", "coral pink"],
        "Light Pink":    ["salmon", "nude pink", "cashmere", "blush", "candy pink", "kashmir",
                          "nude peach", "bright pink", "pink powder", "onion", "kashmeir"],
        "Dark Pink":     ["deep pink", "carmine pink", "dark rose", "fuchsia"],

        # --- Purples ---
        "Purple":        ["plum", "grape"],
        "Light Purple":  ["lilac", "lavender", "violet", "liberty", "mauve", "jacaranda"],
        "Dark Purple":   ["magenta haze", "dark fuchsia", "eggplant", "dark mauve", "magenta", "aubergine"],

        # --- Blues ---
        "Blue":          ["cobalt blue", "sea blue", "azure blue", "cerulean blue", "petrol",
                          "denim", "liberty blue", "citadel blue", "pale blue", "dusky blue",
                          "peacock", "sea port", "peri pop", "sapphire", "petroleum"],
        "Light Blue":    ["sky", "sky blue", "crystal blue", "aqua", "baby blue", "ice",
                          "electric blue", "turquoise", "enamel blue", "nile blue", "blue mint",
                          "cyano", "cyan", "cornflower", "light turquoise", "wild blue",
                          "verdigris", "ice blue", "ginzary", "tiffany"],
        "Dark Blue":     ["navy", "navy blue", "dark denim", "teal", "royal blue", "jeans blue",
                          "deep blue", "indigo", "pepsi blue", "blue navy", "dark teal",
                          "ocean blue", "dark aquamarine", "midnight", "midnight blue",
                          "deep lagoon", "pacific", "dark sky", "dusk blue", "mint blue",
                          "denim blue", "jeans"],

        # --- Greens ---
        "Green":         ["olive", "emerald", "lime", "kiwi", "sage", "pastel green",
                          "emerald green", "dim green", "celadon", "papyrus", "lime green",
                          "lemonade", "limone", "viridescent", "mojito", "seagreen", "jade"],
        "Light Green":   ["mint", "pistache", "apple green", "light mint", "pistachio green",
                          "pistachio", "silver sage", "pistage", "green dolphin",
                          "electric green", "electric lemon", "laurel green"],
        "Dark Green":    ["pine green", "forest green", "dirt green", "moss green",
                          "petrol green", "hunter green", "evergreen", "deep green", "forest",
                          "army", "avocado", "matcha", "army green", "tall evergreen"],

        # --- Beiges ---
        "Beige":         ["greige", "ecru", "dusty blush", "sahara"],
        "Light Beige":   ["sand", "mastic", "cream beige", "off white", "off-white", "ivory beige"],
        "Dark Beige":    ["nude beige", "terracotta gradient"],

        # --- Metallics ---
        "Gold":          ["golden"],
        "Silver":        ["platinum", "pewter", "arsenic"],
        "Bronze":        [],

        # --- Browns ---
        "Brown":         ["dusty brown", "tan", "cafe", "copper", "toffee", "hazel", "cacao",
                          "espresso", "havan", "cappuccino", "chocolate", "cinnamon", "cummin",
                          "hot chocolate", "oxide", "maple"],
        "Light Brown":   ["latte", "kraft", "camel", "mustard", "khaki", "champagne",
                          "coffee nude", "tea", "mocha", "coffee", "chocolate brown",
                          "milk tea", "almond", "blonde", "peru"],
        "Dark Brown":    ["caramel", "walnut", "mocca", "cocoa", "terracotta"],

        # --- Grays ---
        "Gray":          ["grey", "smoke", "paramount", "heather gray", "mortar", "dove", "anthracite"],
        "Light Gray":    ["light grey", "ash", "clay", "stone", "silver grey", "oat",
                          "cloudy grey", "naked concrete", "cloudy silver"],
        "Dark Gray":     ["dark grey", "charcoal", "gunmetal", "charcoal grey", "slate", "taupe", "metal"],

        # --- Black / White / Transparent ---
        "Black":         ["coal", "noir", "midnight black"],
        "White":         ["off white pure", "creamy white", "dusty white", "snow", "creme",
                          "ivory", "white chocolate", "pearl", "alabaster", "powder", "buttermilk"],
        "Transparent":   ["clear", "see through", "neutral"],
    },
# ========================================================================
    "size": {
        "XS":        ["xsmall", "x-small", "x small", "extra small", "wxs"],
        "S":         ["small", "mini", "ys", "wys", "ws", "teeny", "tiny"],
        "M":         ["medium", "med", "ym", "wym", "wm", "mt"],
        "L":         ["large", "big", "yl", "wyl", "wl", "lt"],
        "XL":        ["xlarge", "x-large", "x large", "extra large", "wxl"],
        "XXL":       ["xxlarge", "xx-large", "xx large", "2xl"],
        "One Size":  ["one-size", "onesize", "one"],
        "Free Size": ["free", "free-size", "freesize"],
    },
# ========================================================================
    "age_filter": {
        "2 Years": ["2", "2y", "2 yrs", "2 year", "two years"],
        "3 Years": ["3", "3y", "3 yrs", "3 year", "three years"],
        "4 Years": ["4", "4y", "4 yrs", "4 year", "four years"],
        "5 Years": ["5", "5y", "5 yrs", "5 year", "five years"],
        "6 Years": ["6", "6y", "6 yrs", "6 year", "six years"],
    },
# ========================================================================
    "pattern": {
        "Striped":        ["stripe", "stripes", "lined", "lines"],
        "Dotted":         ["dot", "dots", "polka dot", "polka dots", "spotted", "spots"],
        "Floral":         ["flower", "flowers", "flowery", "floral print"],
        "Plaid":          ["tartan"],
        "Checkered":      ["check", "checks", "checked", "checker"],
        "Printed":        ["plain print", "print", "graphic print", "all-over print"],
        "Argyle":         ["argyll"],
        "Animal Pattern": ["animal print", "leopard", "leopard print", "zebra", "zebra print",
                           "snake", "snake print", "cheetah"],
        "Paisley":        ["paisley print"],
        "Basketweave":    ["basket weave", "woven basket"],
        "Brocade":        ["jacquard"],
        "Camouflage":     ["camo", "camouflaged", "military print"],
        "Embossed":       ["raised pattern", "3d pattern"],
    },
# ========================================================================
    "fashion_type": {
        "Vintage":      ["vintage style", "old school"],
        "Classic":      ["classical", "timeless"],
        "Modern":       ["contemporary", "modernist"],
        "Bohemian":     ["boho", "boho chic", "hippie"],
        "Sporty":       ["sport", "athletic", "athleisure", "active"],
        "Formal":       ["dressy", "business", "office wear"],
        "Gothic":       ["goth"],
        "Ethnic":       ["traditional", "tribal", "folk"],
        "Casual":       ["everyday", "daily", "informal"],
        "Street":       ["streetwear", "urban", "hip hop"],
        "Grunge":       ["grungy"],
        "Punk":         ["punk rock"],
        "Artistic":     ["artsy", "art style"],
        "Preppy":       ["prep", "ivy league"],
        "Retro":        ["retro style", "throwback"],
        "Beachwear":    ["beach", "swimwear", "resort wear"],
        "Minimalist":   ["minimal", "simple", "clean"],
        "Evening Wear": ["evening", "cocktail", "party wear", "gala"],
    },
# ========================================================================
    "material": {
        "Puff":    ["puffy", "puffer"],
        "Natural": ["natural fiber", "organic"],
        "Blend":   ["blended", "mixed fiber"],
        "Knit":    ["knitted", "knitwear"],
        "Fur":     ["faux fur", "furry"],
        "Leather": ["faux leather", "pu leather", "pleather", "genuine leather"],
        "Linen":   ["linen blend"],
        "Teddy":   ["teddy bear", "teddy fabric", "sherpa"],
        "Wood":    ["wooden", "timber"],
        "Plush":   ["plushy", "velour", "soft plush"],
    },
# ========================================================================
    "scent": {
        "Wood":   ["woody", "wooden", "sandalwood", "cedar scent"],
        "Sweet":  ["sugary", "vanilla", "caramel scent", "candy"],
        "Fruity": ["fruit", "citrus", "berry", "tropical fruit"],
        "Floral": ["flower", "flowery", "rose scent", "jasmine"],
    },
# ========================================================================
    "purpose": {
        "Decoration": ["decor", "decorative", "ornament", "ornamental", "display"],
    },
# ========================================================================
    "gender": {
        "Men":                         ["male", "man", "gents", "gentleman", "mens"],
        "Women":                       ["female", "woman", "ladies", "lady", "womens"],
        "Unisex women, Unisex men":    ["unisex", "uni", "for all", "both",
                                        "unisex adults", "unisex adult", "unisex woman, unisex man"],
        "Boys":                        ["boy", "kids boys", "young boys"],
        "Girls":                       ["girl", "kids girls", "young girls"],
        "Unisex girls, Unisex boys":   ["unisex kids", "unisex children", "unisex child",
                                        "unisex baby", "unisex babies",
                                        "unisex girl, unisex boy"],
    },
# ========================================================================
    "feature": {
        "Jewelry":         ["jewellery", "accessories jewelry"],
        "Clothes":         ["clothing", "apparel", "garment", "garments"],
        "Footwear":        ["shoes", "shoe", "sneakers", "boots"],
        "Bags":            ["bag", "handbag", "purse", "tote"],
        "Glasses":         ["eyewear", "sunglasses", "spectacles"],
        "Beauty":          ["cosmetics", "makeup", "skincare"],
        "Kids":            ["children", "kid", "baby items"],
        "Structure":       ["structured", "rigid"],
        "Gold Plate":      ["gold plated", "gold-plated", "gold finish"],
        "Waterproof":      ["water proof", "water-proof", "fully waterproof"],
        "Handmade":        ["hand made", "hand-made", "artisanal"],
        "Embroidered":     ["embroidery", "stitched design"],
        "Water Repellent": ["water-repellent", "water resistant", "water-resistant"],
        "UV Protection":   ["uv protected", "uv-protection", "anti uv", "sun protection"],
        "Fast Dry":        ["fast-dry", "quick dry", "quick-dry", "rapid dry"],
        "Stretchy":        ["stretch", "elastic", "flexible"],
        "Lightweight":     ["light weight", "light-weight", "ultra light"],
        "Anti-bacterial":  ["antibacterial", "anti bacterial", "antimicrobial"],
        "Sweat Repellent": ["sweat-repellent", "sweat resistant", "moisture wicking", "wicking"],
        "Lined":           ["with lining", "lining"],
        "Slip-on":         ["slip on", "slipon", "easy on"],
        "Handcrafted":     ["hand crafted", "hand-crafted", "artisan crafted"],
    },
# ========================================================================
    "season": {
        "Summer": ["summery", "hot weather", "warm season"],
        "Spring": ["springtime", "spring season"],
        "Fall":   ["autumn", "fall season"],
        "Winter": ["wintertime", "cold weather", "cold season"],
    },
# ========================================================================
    "decor_style": {
        "Islamic":  ["arabic", "arabesque", "moroccan", "oriental islamic"],
        "Modern":   ["contemporary", "modernist"],
        "Rustic":   ["country", "farmhouse", "vintage rustic"],
        "Minimal":  ["minimalist", "simple", "clean"],
        "Bohemian": ["boho", "boho chic", "hippie"],
    },
# ========================================================================
    "month": {
        "January":   ["jan", "1", "01"],
        "February":  ["feb", "2", "02"],
        "March":     ["mar", "3", "03"],
        "April":     ["apr", "4", "04"],
        "May":       ["5", "05"],
        "June":      ["jun", "6", "06"],
        "July":      ["jul", "7", "07"],
        "August":    ["aug", "8", "08"],
        "September": ["sep", "sept", "9", "09"],
        "October":   ["oct", "10"],
        "November":  ["nov", "11"],
        "December":  ["dec", "12"],
    },
# ========================================================================
    "birthstone": {
        "Garnet":        ["garnet stone", "red garnet"],
        "Amethyst":      ["amethyst stone", "purple amethyst"],
        "Aquamarine":    ["aqua marine", "aqua-marine"],
        "White Topaz":   ["white-topaz", "topaz white"],
        "Green Onyx":    ["green-onyx", "onyx green"],
        "Moonstone":     ["moon stone", "moon-stone"],
        "Ruby":          ["ruby stone", "red ruby"],
        "Peridot":       ["peridot stone"],
        "Blue Sapphire": ["blue-sapphire", "sapphire blue", "sapphire"],
        "Opal":          ["opal stone"],
        "Citrine":       ["citrine stone", "yellow citrine"],
        "Blue Topaz":    ["blue-topaz", "topaz blue"],
    },
# ========================================================================
}
