import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from verify_logo import LogoForensicsEngine

engine = LogoForensicsEngine()
test_path = project_root / "tests" / "fixtures" / "07_changed_number_a500m.png"
if not test_path.exists():
    print("Fixture does not exist, creating...")
    from tests.test_forensics_suite import TestForensicsEngine
    TestForensicsEngine.setUpClass()

res = engine.verify(test_path, skip_vlm=True)
print("="*60)
print(f"VERDICT: {res['verdict']} | THREAT: {res['threat_type']}")
print(f"BRAND: {res['brand_id']} | MATCHED REF: {res['matched_reference']}")
print(f"METRICS: {res['forensic_metrics']}")
print(f"OCR: {res['ocr']}")
print(f"DIFFS: {res['micro_differences_detected']}")
print(f"REASON: {res['action_reason']}")
