import os
import torch
import re


# ============================================================
# CONFIG
# ============================================================

DATABASE_PATH = "reference_embeddings/ocr_database.pt"


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("OCR DATABASE VALIDATION")
print("=" * 70)


# ============================================================
# LOAD DATABASE
# ============================================================

if not os.path.exists(DATABASE_PATH):
    print(f"[ERROR] Database not found:")
    print(f"        {DATABASE_PATH}")
    raise SystemExit(1)


database = torch.load(
    DATABASE_PATH,
    map_location="cpu",
    weights_only=False
)

print("Database loaded successfully.")
print()


# ============================================================
# DATABASE STRUCTURE
# ============================================================

if isinstance(database, dict):

    print("Database type: dictionary")
    print()

    print("Top-level keys:")

    for key in database.keys():
        value = database[key]

        if isinstance(value, torch.Tensor):
            print(f"  {key:<20} Tensor {tuple(value.shape)}")

        elif isinstance(value, dict):
            print(f"  {key:<20} dict ({len(value)} items)")

        elif isinstance(value, list):
            print(f"  {key:<20} list ({len(value)} items)")

        else:
            print(f"  {key:<20} {type(value).__name__}")

else:

    print(f"Database type: {type(database).__name__}")


print()


# ============================================================
# FIND BRAND DATA
# ============================================================

brands = None

if isinstance(database, dict):

    for key in [
        "brands",
        "database",
        "references",
        "ocr_database",
        "entries"
    ]:

        if key in database and isinstance(database[key], dict):
            brands = database[key]
            break

if brands is None and isinstance(database, dict):

    # Detect dictionary containing brand-like entries
    possible = {}

    for key, value in database.items():

        if isinstance(value, dict):

            possible[key] = value

    if possible:
        brands = possible


if brands is None:

    print("[WARNING] Could not automatically identify brand entries.")

    print()
    print("Raw database structure:")
    print(database)

    raise SystemExit(0)


# ============================================================
# BRAND SUMMARY
# ============================================================

print("=" * 70)
print("BRAND SUMMARY")
print("=" * 70)

print(f"Brands found: {len(brands)}")
print()


# ============================================================
# INSPECT BRANDS
# ============================================================

total_texts = 0
brands_with_text = 0
brands_without_text = 0


for brand_name, data in sorted(brands.items()):

    texts = []

    if isinstance(data, dict):

        for key in [
            "texts",
            "ocr_texts",
            "candidates",
            "results"
        ]:

            if key in data:

                value = data[key]

                if isinstance(value, list):
                    texts = value

                elif isinstance(value, dict):
                    texts = list(value.keys())

                break

    if texts:

        brands_with_text += 1
        total_texts += len(texts)

        print(f"{brand_name:<18} -> {len(texts):>3} OCR entries")

    else:

        brands_without_text += 1

        print(f"{brand_name:<18} ->   0 OCR entries")


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Total brands:        {len(brands)}")
print(f"Brands with OCR:     {brands_with_text}")
print(f"Brands without OCR:  {brands_without_text}")
print(f"Total OCR entries:   {total_texts}")


# ============================================================
# VALIDATION CHECKS
# ============================================================

print()
print("=" * 70)
print("VALIDATION CHECKS")
print("=" * 70)

checks = 0
passed = 0


def check(name, condition):
    global checks, passed

    checks += 1

    if condition:
        print(f"[PASS] {name}")
        passed += 1
    else:
        print(f"[FAIL] {name}")


check(
    "Database file exists",
    os.path.exists(DATABASE_PATH)
)

check(
    "Database loaded",
    database is not None
)

check(
    "Brand entries found",
    len(brands) > 0
)

check(
    "Multiple brands present",
    len(brands) >= 2
)

check(
    "OCR data exists",
    brands_with_text > 0
)


# ============================================================
# CHECK EXPECTED BRAND COUNT
# ============================================================

check(
    "Expected 52 brands",
    len(brands) == 52
)


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 70)
print("VALIDATION RESULT")
print("=" * 70)

print(f"Checks passed: {passed}/{checks}")

if passed == checks:
    print("OCR DATABASE VALIDATION PASSED")
else:
    print("OCR DATABASE VALIDATION NEEDS REVIEW")

print("=" * 70)