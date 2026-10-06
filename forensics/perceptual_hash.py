"""
Perceptual Hashing Module.
Computes pHash (Frequency DCT) and dHash (Gradient Difference) for ultra-fast screening.
Note: A Hamming distance of 0 indicates perceptual similarity, NOT 100% byte identity.
"""

from typing import Dict, Any, Union
import numpy as np
import cv2
from PIL import Image


def compute_dhash(image: Image.Image, hash_size: int = 8) -> str:
    """
    Computes difference hash (dHash) by comparing adjacent pixels in a gradient row.
    Produces a (hash_size * hash_size)-bit binary hex string.
    """
    # Resize to (hash_size + 1, hash_size)
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    arr = np.array(gray, dtype=np.int32)
    # Compare adjacent pixels
    diff = arr[:, 1:] > arr[:, :-1]
    # Flatten to binary string
    bits = "".join(["1" if b else "0" for b in diff.flatten()])
    # Convert to hex string
    hex_str = f"{int(bits, 2):0{hash_size * hash_size // 4}x}"
    return hex_str


def compute_phash(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    """
    Computes DCT-based perceptual hash (pHash).
    Extracts low-frequency DCT components representing coarse visual geometry.
    """
    img_size = hash_size * highfreq_factor
    gray = image.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)
    arr = np.array(gray, dtype=np.float32)

    # Compute 2D DCT
    dct = cv2.dct(arr)

    # Extract top-left 8x8 low frequency coefficients (excluding DC term at [0, 0])
    dct_low = dct[:hash_size, :hash_size]
    med = np.median(dct_low[1:, 1:]) if hash_size > 1 else np.median(dct_low)

    diff = dct_low > med
    bits = "".join(["1" if b else "0" for b in diff.flatten()])
    hex_str = f"{int(bits, 2):0{hash_size * hash_size // 4}x}"
    return hex_str


def hamming_distance(hex1: str, hex2: str) -> int:
    """
    Calculates bitwise Hamming distance between two hex hashes.
    """
    if not hex1 or not hex2:
        return 64
    try:
        val1 = int(hex1, 16)
        val2 = int(hex2, 16)
        return bin(val1 ^ val2).count("1")
    except Exception:
        return 64


class PerceptualHasher:
    @staticmethod
    def extract_hashes(image: Image.Image) -> Dict[str, str]:
        """
        Extracts both pHash and dHash for an image.
        """
        return {
            "phash": compute_phash(image),
            "dhash": compute_dhash(image),
        }

    @staticmethod
    def compare_hashes(
        query_hashes: Dict[str, str], ref_hashes: Dict[str, str]
    ) -> Dict[str, int]:
        """
        Computes Hamming distances between query and reference hashes.
        """
        p_dist = hamming_distance(query_hashes.get("phash", ""), ref_hashes.get("phash", ""))
        d_dist = hamming_distance(query_hashes.get("dhash", ""), ref_hashes.get("dhash", ""))
        return {
            "phash_hamming_distance": p_dist,
            "dhash_hamming_distance": d_dist,
        }
