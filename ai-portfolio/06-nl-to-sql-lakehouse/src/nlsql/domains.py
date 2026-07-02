"""Canonical categorical domains for the star schema.

These value lists are the single source shared by the data generator (which
samples from them) and the schema linker (which recognises them in natural
language, e.g. mapping the token "Electronics" to a ``product_category`` filter).
Keeping them here guarantees the generator and the NL parser can never drift.
"""
from __future__ import annotations

# dim_customer
SEGMENTS = ["Consumer", "Corporate", "SMB"]
CUSTOMER_COUNTRIES = ["USA", "Canada", "UK", "Germany", "France"]

# dim_product
CATEGORIES = ["Electronics", "Furniture", "Office Supplies", "Apparel", "Grocery"]
SUBCATEGORIES = {
    "Electronics": ["Phones", "Laptops", "Accessories"],
    "Furniture": ["Chairs", "Tables", "Storage"],
    "Office Supplies": ["Paper", "Binders", "Pens"],
    "Apparel": ["Shirts", "Shoes", "Outerwear"],
    "Grocery": ["Snacks", "Beverages", "Produce"],
}
BRANDS = ["Acme", "Globex", "Initech", "Umbrella", "Soylent", "Hooli"]

# dim_store
REGIONS = ["North", "South", "East", "West", "Central"]
STORE_TYPES = ["Retail", "Online", "Outlet"]
STORE_COUNTRIES = ["USA", "Canada", "UK", "Germany", "France"]

# dim_date
YEARS = [2021, 2022, 2023]

# Reverse map: filter value (lowercased) -> (dimension_name, canonical_value)
# Used by the schema linker to recognise literal filter values in a question.
VALUE_TO_DIMENSION: dict[str, tuple[str, str]] = {}
for _v in SEGMENTS:
    VALUE_TO_DIMENSION[_v.lower()] = ("customer_segment", _v)
for _v in CATEGORIES:
    VALUE_TO_DIMENSION[_v.lower()] = ("product_category", _v)
for _v in BRANDS:
    VALUE_TO_DIMENSION[_v.lower()] = ("product_brand", _v)
for _v in REGIONS:
    VALUE_TO_DIMENSION[_v.lower()] = ("store_region", _v)
for _v in STORE_TYPES:
    VALUE_TO_DIMENSION[_v.lower()] = ("store_type", _v)
for _v in CUSTOMER_COUNTRIES:
    VALUE_TO_DIMENSION[_v.lower()] = ("customer_country", _v)
