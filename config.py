"""
Central Configuration for Multi-Stage Logo Recognition System.
Loads settings from config.yaml with hardcoded fallback defaults.
All paths, model identifiers, score fusion weights, and decision thresholds
are centralized here.

Existing code uses: config.SIGLIP_WEIGHT, config.VLM_URL, etc.
This module preserves that interface while reading from YAML.
"""

from pathlib import Path

# ============================================================
# YAML LOADER
# ============================================================

def _load_yaml_config() -> dict:
    """Load config.yaml from project root, return empty dict on failure."""
    yaml_path = Path(__file__).resolve().parent / "config.yaml"
    if not yaml_path.exists():
        return {}
    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # PyYAML not installed — parse simple YAML manually as fallback
        # For production, install PyYAML: pip install pyyaml
        return {}
    except Exception:
        return {}

_cfg = _load_yaml_config()

def _get(section: str, key: str, default):
    """Safely get a nested value from the YAML config."""
    sec = _cfg.get(section, {})
    if isinstance(sec, dict):
        return sec.get(key, default)
    return default


# ============================================================
# DIRECTORY & FILE PATHS (always computed, not from YAML)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
LOGOS_DIR = BASE_DIR / "logos"
FAVICON_DIR = BASE_DIR / "Favicon"
REFERENCE_DIR = BASE_DIR / "reference_embeddings"
REFERENCE_EMBEDDINGS_DIR = REFERENCE_DIR

BRAND_DB_PATH = REFERENCE_DIR / "brand_database.pt"
BRAND_MULTI_DB_PATH = REFERENCE_DIR / "brand_multi_database.pt"
COLOR_DB_PATH = REFERENCE_DIR / "color_database.pt"
OCR_DB_PATH = REFERENCE_DIR / "ocr_database.pt"
BRAND_LIST_PATH = REFERENCE_DIR / "brand_list.txt"
UNIFIED_DB_PATH = REFERENCE_DIR / "unified_brand_database.pt"

DATASET_DIR = BASE_DIR / "dataset"
OUR_LOGOS_DIR = DATASET_DIR / "our_logos"
COMPETITOR_LOGOS_DIR = DATASET_DIR / "competitor_logos"


# ============================================================
# MODEL CONFIGURATION
# ============================================================

SIGLIP_MODEL_NAME = _get("model", "siglip_model_name", "google/siglip2-so400m-patch14-384")

# Versioned reference embedding directory and file paths
def get_siglip_slug(model_name: str) -> str:
    slug = model_name.replace("/", "_").replace("-", "_")
    short_slug = slug.replace("google_", "")
    if (REFERENCE_DIR / short_slug).exists() and not (REFERENCE_DIR / slug).exists():
        return short_slug
    return slug

SIGLIP_MODEL_SLUG = get_siglip_slug(SIGLIP_MODEL_NAME)
if (REFERENCE_DIR / "siglip2_so400m_patch14_384").exists() and "so400m" in SIGLIP_MODEL_NAME.lower():
    SIGLIP_VERSION_DIR = REFERENCE_DIR / "siglip2_so400m_patch14_384"
elif (REFERENCE_DIR / "siglip2_base_patch16_224").exists() and "base" in SIGLIP_MODEL_NAME.lower():
    SIGLIP_VERSION_DIR = REFERENCE_DIR / "siglip2_base_patch16_224"
else:
    SIGLIP_VERSION_DIR = REFERENCE_DIR / SIGLIP_MODEL_SLUG

FORENSIC_REF_STORE_PATH = SIGLIP_VERSION_DIR / "forensic_reference_store.pt"

# Preprocessing views to generate for visual ensemble
PREPROCESSING_VIEWS = _cfg.get("preprocessing_views", [
    "raw_rgb",
    "original",
    "resized",
    "square_padded",
    "center_crop",
    "contrast_enhanced",
    "white_bg",
    "black_bg",
    "sharpened",
])

# Ensemble strategy for SigLIP multi-view embeddings: 'max', 'weighted_top_k', 'mean_top_k'
SIGLIP_ENSEMBLE_STRATEGY = _get("model", "siglip_ensemble_strategy", "max")
SIGLIP_TOP_K_VIEWS = _get("model", "siglip_top_k_views", 3)

# View weights for weighted ensemble
VIEW_WEIGHTS = _cfg.get("view_weights", {
    "raw_rgb": 1.0,
    "original": 0.95,
    "square_padded": 0.90,
    "white_bg": 0.85,
    "black_bg": 0.85,
    "contrast_enhanced": 0.80,
    "resized": 0.75,
    "center_crop": 0.70,
    "sharpened": 0.65,
})

# ============================================================
# SCORE FUSION WEIGHTS
# ============================================================

SIGLIP_WEIGHT = _get("fusion", "siglip_weight", 0.50)
COLOR_WEIGHT = _get("fusion", "color_weight", 0.20)
OCR_WEIGHT = _get("fusion", "ocr_weight", 0.15)
CLASSIFIER_WEIGHT = _get("fusion", "classifier_weight", 0.15)
VLM_WEIGHT = _get("fusion", "vlm_weight", 0.00)
STRUCTURE_WEIGHT = _get("fusion", "structure_weight", 0.00)

# ============================================================
# BRAND CLASSIFIER CONFIGURATION (Phase 9)
# ============================================================
CLASSIFIER_ENABLED = _get("classifier", "enabled", True)
CLASSIFIER_MODEL_PATH = REFERENCE_DIR / "brand_classifier.pt"
CLASSIFIER_UNKNOWN_THRESHOLD = _get("classifier", "unknown_threshold", 0.12)

# Neutral score assigned when an image contains no readable text
OCR_NEUTRAL_SCORE = _get("ocr", "neutral_score", 0.50)

# Minimum OCR text length and confidence for matching
OCR_MIN_CONFIDENCE = _get("ocr", "min_confidence", 0.25)
OCR_MIN_CHAR_LENGTH = _get("ocr", "min_char_length", 2)

# ============================================================
# DECISION THRESHOLDS (MATCH, REVIEW, UNKNOWN)
# ============================================================

# Confident match requires both high score and significant margin over runner-up
MATCH_SCORE_THRESHOLD = _get("decision", "match_score_threshold", 0.80)
MATCH_MARGIN_THRESHOLD = _get("decision", "match_margin_threshold", 0.030)

# Moderate confidence or small margin triggers human/system REVIEW
REVIEW_SCORE_THRESHOLD = _get("decision", "review_score_threshold", 0.65)
REVIEW_MARGIN_THRESHOLD = _get("decision", "review_margin_threshold", 0.010)

# Below REVIEW_SCORE_THRESHOLD is classified as UNKNOWN

# Evidence-aware thresholds (Phase 5)
STRONG_VISUAL_THRESHOLD = _get("evidence_rules", "strong_visual_threshold", 0.85)
STRONG_COLOR_THRESHOLD = _get("evidence_rules", "strong_color_threshold", 0.80)
STRONG_OCR_THRESHOLD = _get("evidence_rules", "strong_ocr_threshold", 0.80)
CONFLICT_MARGIN = _get("evidence_rules", "conflict_margin", 0.15)

# Section 18 Model-Specific Forensic Thresholds (Calibrated for SigLIP 2 SO400M)
SIGLIP_MATCH_THRESHOLD = _get("siglip", "strong_match", 0.92)
SIGLIP_REVIEW_THRESHOLD = _get("siglip", "suspicious", 0.83)
DINO_MATCH_THRESHOLD = _get("dinov2", "strong_match", 0.88)
COLOR_DELTAE_THRESHOLD = _get("color", "delta_e_match", 3.5)
EDGE_IOU_THRESHOLD = _get("edge", "strong_match", 0.72)
CLASSIFIER_THRESHOLD = _get("classifier", "confidence_threshold", 0.80)

# ============================================================
# PROTECTED BRANDS CONFIGURATION (Zero False Negatives Target)
# ============================================================

# Brands requiring extra-conservative rejection and sensitive matching
_protected_cfg = _cfg.get("protected_brands", {})
PROTECTED_BRANDS = _protected_cfg.get("brands", ["a200m"]) if isinstance(_protected_cfg, dict) else ["a200m"]

# For protected brands, lower the threshold to trigger REVIEW rather than UNKNOWN
PROTECTED_SCORE_THRESHOLD = _get("protected_brands", "score_threshold", 0.60)
PROTECTED_MARGIN_THRESHOLD = _get("protected_brands", "margin_threshold", 0.015)
PROTECTED_VLM_TRIGGER_MARGIN = _get("protected_brands", "vlm_trigger_margin", 0.040)

# ============================================================
# VLM FALLBACK CONFIGURATION (Ambiguous Cases Only)
# ============================================================

VLM_ENABLED = _get("vlm", "enabled", True)
VLM_URL = _get("vlm", "url", "http://localhost:11434/api/generate")
VLM_MODEL = _get("vlm", "model", "qwen2.5vl:7b")
VLM_TIMEOUT_SECONDS = _get("vlm", "timeout_seconds", 30)
VLM_AMBIGUITY_MARGIN = _get("vlm", "ambiguity_margin", 0.025)
VLM_TOP_CANDIDATES = _get("vlm", "top_candidates", 4)

# ============================================================
# MULTI-STAGE PIPELINE CONFIGURATION (Phase 6)
# ============================================================

FAST_PATH_SCORE = _get("pipeline", "fast_path_score", 0.92)
FAST_PATH_MARGIN = _get("pipeline", "fast_path_margin", 0.08)
VLM_TRIGGER_ON_DISAGREEMENT = _get("pipeline", "vlm_trigger_on_disagreement", True)
VLM_TRIGGER_ON_AMBIGUITY = _get("pipeline", "vlm_trigger_on_ambiguity", True)
VLM_TRIGGER_ON_PROTECTED = _get("pipeline", "vlm_trigger_on_protected", True)

# ============================================================
# LOGGING CONFIGURATION
# ============================================================

LOG_LEVEL = _get("logging", "level", "INFO")
LOG_PER_IMAGE = _get("logging", "per_image_breakdown", True)
LOG_FILE = _get("logging", "log_file", None)
