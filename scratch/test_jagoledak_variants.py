import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from verify_logo import LogoForensicsEngine
from forensics.color_analysis import ColorAnalyzer
from preprocessing.media_normalizer import MediaNormalizer
from PIL import Image

engine = LogoForensicsEngine()
q_path = 'dataset/our_logos/raw_logos/jagoledak.webp'
norm = MediaNormalizer.normalize_media(q_path)
q_color = ColorAnalyzer.extract_color_profile(norm['normalized_rgb'], alpha_mask=norm['alpha_mask'])

ref = engine.ref_store.references['jagoledak']
variants = ref.get('variants', [ref])
for i, v in enumerate(variants):
    comp = ColorAnalyzer.compare_color_profiles(q_color, v['color_profile'])
    fn = v.get('filename')
    print(f"{i}: {fn} -> Delta E = {comp['cielab_delta_e']}, Hist = {comp['histogram_similarity']}")
