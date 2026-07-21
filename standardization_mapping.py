# ------------------------------------------------------------------------------------------------------------------------------ #
# Color
color = [
    "Yellow", "Light Yellow", "Dark Yellow",
    "Orange", "Light Orange", "Dark Orange",
    "Red", "Light Red", "Dark Red",
    "Pink", "Light Pink", "Dark Pink",
    "Purple", "Light Purple", "Dark Purple",
    "Blue", "Light Blue", "Dark Blue",
    "Green", "Light Green", "Dark Green",
    "Beige", "Light Beige", "Dark Beige",
    "Gold", "Silver", "Bronze",
    "Brown", "Light Brown", "Dark Brown",
    "Gray", "Light Gray", "Dark Gray",
    "Black", "White", "Transparent",
]
# ------------------------------------------------------------------------------------------------------------------------------ #
# Size
size = ["XS", "S", "M", "L", "XL", "XXL", "One Size", "Free Size"]
# ------------------------------------------------------------------------------------------------------------------------------ #
# Age Filter
age_filter = [
    # Months (baby / toddler)
    "0 Months", "1 Months", "2 Months", "3 Months", "4 Months", "5 Months",
    "6 Months", "7 Months", "8 Months", "9 Months", "10 Months", "11 Months",
    "12 Months", "18 Months", "24 Months",
    # Years
    "1 Years", "2 Years", "3 Years", "4 Years", "5 Years",
    "6 Years", "7 Years", "8 Years", "9 Years", "10 Years",
    "11 Years", "12 Years", "13 Years", "14 Years",
]
# ------------------------------------------------------------------------------------------------------------------------------ #
# Pattern
pattern = [
    "Striped", "Dotted", "Floral", "Plaid", "Checkered", "Printed",
    "Argyle", "Animal Pattern", "Paisley", "Basketweave", "Brocade",
    "Camouflage", "Embossed",
]
# ------------------------------------------------------------------------------------------------------------------------------ #
# Fashion Type
fashion_type = [
    "Vintage", "Classic", "Modern", "Bohemian", "Sporty", "Formal",
    "Gothic", "Ethnic", "Casual", "Street", "Grunge", "Punk",
    "Artistic", "Preppy", "Retro", "Beachwear", "Minimalist", "Evening Wear",
]
# ------------------------------------------------------------------------------------------------------------------------------ #
# Material
material = [
    "Puff", "Natural", "Blend", "Knit", "Fur", "Leather",
    "Linen", "Teddy", "Wood", "Plush",
]
# ------------------------------------------------------------------------------------------------------------------------------ #
# Scent
scent = ["Wood", "Sweet", "Fruity", "Floral"]
# ------------------------------------------------------------------------------------------------------------------------------ #
# Purpose
purpose = ["Decoration"]
# ------------------------------------------------------------------------------------------------------------------------------ #
# Gender
gender = ["Men", "Women", "Unisex women, Unisex men", "Boys", "Girls", "Unisex girls, Unisex boys"]
# ------------------------------------------------------------------------------------------------------------------------------ #
# Feature
feature = [
    "Jewelry", "Clothes", "Footwear", "Bags", "Glasses", "Beauty", "Kids", "Structure",
    "Gold Plate", "Waterproof", "Handmade", "Embroidered", "Water Repellent",
    "UV Protection", "Fast Dry", "Stretchy", "Lightweight", "Anti-bacterial",
    "Sweat Repellent", "Lined", "Slip-on", "Handcrafted",
]
# ------------------------------------------------------------------------------------------------------------------------------ #
# Season
season = ["Summer", "Spring", "Fall", "Winter"]
# ------------------------------------------------------------------------------------------------------------------------------ #
# Decor Style
decor_style = ["Islamic", "Modern", "Rustic", "Minimal", "Bohemian"]
# ------------------------------------------------------------------------------------------------------------------------------ #
# Month
month = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
# ------------------------------------------------------------------------------------------------------------------------------ #
# Birthstone
birthstone = [
    "Garnet", "Amethyst", "Aquamarine", "White Topaz", "Green Onyx",
    "Moonstone", "Ruby", "Peridot", "Blue Sapphire", "Opal", "Citrine", "Blue Topaz",
]
# ------------------------------------------------------------------------------------------------------------------------------ #
# Master map — key is the attribute name, value is the allowed list
standardization_map = {
    "color": color,
    "size": size,
    "age_filter": age_filter,
    "age": age_filter,  # alias — templates use the plain "Age" field name
    "pattern": pattern,
    "fashion_type": fashion_type,
    "material": material,
    "scent": scent,
    "purpose": purpose,
    "gender": gender,
    "feature": feature,
    "season": season,
    "decor_style": decor_style,
    "month": month,
    "birthstone": birthstone,
}
