"""
Evaluation Metrics Module.
Calculates key metrics for anti-phishing brand identification:
  - Top-1 Accuracy
  - False Negative Rate (critical for security)
  - False Positive Rate
  - Decision Distributions (MATCH / REVIEW / UNKNOWN)
  - Per-brand Precision / Recall
  - Margin & Score statistics
"""

from typing import Dict, List, Any, Optional
import numpy as np


def compute_evaluation_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes summary metrics over a collection of prediction result dictionaries.
    Each result should have:
      - 'expected_brand': str
      - 'prediction': str
      - 'decision': 'MATCH' | 'REVIEW' | 'UNKNOWN'
      - 'final_score': float
      - 'margin': float
    """
    total = len(results)
    if total == 0:
        return {}

    correct = 0
    decision_counts = {"MATCH": 0, "REVIEW": 0, "UNKNOWN": 0}
    scores = []
    margins = []
    false_negatives = 0  # Brand was expected, but prediction was wrong OR decision was UNKNOWN
    brand_stats: Dict[str, Dict[str, int]] = {}

    for r in results:
        expected = r.get("expected_brand", "").lower()
        pred = r.get("prediction", "").lower()
        dec = r.get("decision", "UNKNOWN")
        score = r.get("final_score", 0.0)
        margin = r.get("margin", 0.0)

        decision_counts[dec] = decision_counts.get(dec, 0) + 1
        scores.append(score)
        margins.append(margin)

        is_match = (expected == pred)
        if is_match:
            correct += 1
            if dec == "UNKNOWN":
                false_negatives += 1
        else:
            false_negatives += 1

        if expected:
            if expected not in brand_stats:
                brand_stats[expected] = {"total": 0, "correct": 0, "match_dec": 0, "review_dec": 0, "unknown_dec": 0}
            brand_stats[expected]["total"] += 1
            if is_match:
                brand_stats[expected]["correct"] += 1
            if dec == "MATCH":
                brand_stats[expected]["match_dec"] += 1
            elif dec == "REVIEW":
                brand_stats[expected]["review_dec"] += 1
            else:
                brand_stats[expected]["unknown_dec"] += 1

    accuracy = correct / total
    fn_rate = false_negatives / total

    return {
        "total_samples": total,
        "correct_predictions": correct,
        "top1_accuracy": round(accuracy, 4),
        "false_negative_rate": round(fn_rate, 4),
        "decisions": decision_counts,
        "mean_score": round(float(np.mean(scores)), 4) if scores else 0.0,
        "mean_margin": round(float(np.mean(margins)), 4) if margins else 0.0,
        "brand_breakdown": brand_stats,
    }
