from pathlib import Path
from PIL import Image, ImageSequence, ImageChops, ImageStat
import numpy as np

p1 = Path("test_images/11.gif")
p2 = Path("logos/dewi11.gif")

print(f"p1 exists: {p1.exists()}, size: {p1.stat().st_size} bytes")
print(f"p2 exists: {p2.exists()}, size: {p2.stat().st_size} bytes")

im1 = Image.open(p1)
im2 = Image.open(p2)

n_frames1 = getattr(im1, "n_frames", 1)
n_frames2 = getattr(im2, "n_frames", 1)

print(f"im1 (test_images/11.gif): format={im1.format}, size={im1.size}, mode={im1.mode}, frames={n_frames1}")
print(f"im2 (logos/dewi11.gif)    : format={im2.format}, size={im2.size}, mode={im2.mode}, frames={n_frames2}")

# Frame durations
durations1 = [frame.info.get('duration', 0) for frame in ImageSequence.Iterator(im1)]
durations2 = [frame.info.get('duration', 0) for frame in ImageSequence.Iterator(im2)]
print(f"im1 frame durations (first 10): {durations1[:10]}, total duration: {sum(durations1)}ms")
print(f"im2 frame durations (first 10): {durations2[:10]}, total duration: {sum(durations2)}ms")

# Frame comparison
im1.seek(0)
im2.seek(0)
f1_0 = im1.convert("RGBA")
f2_0 = im2.convert("RGBA")

print(f"Frame 0 sizes: f1={f1_0.size}, f2={f2_0.size}")

# Color palette / stats
stat1 = ImageStat.Stat(f1_0)
stat2 = ImageStat.Stat(f2_0)
print(f"Frame 0 mean RGBA: im1={stat1.mean}, im2={stat2.mean}")

# Compare frames
diff_count = 0
max_diff = 0
min_frames = min(n_frames1, n_frames2)
for idx in range(min_frames):
    im1.seek(idx)
    im2.seek(idx)
    fr1 = im1.convert("RGBA")
    fr2 = im2.convert("RGBA")
    if fr1.size != fr2.size:
        fr2 = fr2.resize(fr1.size)
    diff = ImageChops.difference(fr1, fr2)
    bbox = diff.getbbox()
    if bbox:
        diff_count += 1
        stat = ImageStat.Stat(diff)
        d_mean = sum(stat.mean) / len(stat.mean)
        if d_mean > max_diff:
            max_diff = d_mean

print(f"Frames compared: {min_frames}")
print(f"Number of frames with differences: {diff_count}")
print(f"Max average pixel difference across channels: {max_diff:.2f}")

# Save first frames and difference image to scratch
scratch_dir = Path("scratch")
scratch_dir.mkdir(exist_ok=True)
f1_0.save(scratch_dir / "11_frame0.png")
f2_0.save(scratch_dir / "dewi11_frame0.png")
diff_0 = ImageChops.difference(f1_0, f2_0.resize(f1_0.size))
diff_0.save(scratch_dir / "diff_frame0.png")
print("Saved 11_frame0.png, dewi11_frame0.png, and diff_frame0.png to scratch/")
