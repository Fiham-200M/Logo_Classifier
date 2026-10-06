from PIL import Image
from pathlib import Path

IMAGE_DIR = Path(__file__).resolve().parent / "logos"

files = [
    "our_logo1.png",
    "our_logo1_dark.png",
    "our_logo1_small.png",
]

print("=" * 80)
print("A200M LOGO GEOMETRY / ALPHA ANALYSIS")
print("=" * 80)

for filename in files:
    path = IMAGE_DIR / filename
    image = Image.open(path).convert("RGBA")

    width, height = image.size
    pixels = image.load()

    transparent = 0
    nontransparent = 0
    nonwhite = 0

    xs = []
    ys = []

    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]

            if a == 0:
                transparent += 1
            else:
                nontransparent += 1

                if not (r > 245 and g > 245 and b > 245):
                    nonwhite += 1
                    xs.append(x)
                    ys.append(y)

    total = width * height

    print()
    print(f"FILE:              {filename}")
    print(f"Dimensions:        {width} x {height}")
    print(f"Aspect ratio:      {width / height:.4f}")
    print(f"Transparent:       {transparent / total * 100:.2f}%")
    print(f"Non-transparent:   {nontransparent / total * 100:.2f}%")
    print(f"Non-white:         {nonwhite / total * 100:.2f}%")

    if xs:
        left = min(xs)
        top = min(ys)
        right = max(xs) + 1
        bottom = max(ys) + 1

        bbox_width = right - left
        bbox_height = bottom - top

        print(f"Content bbox:      ({left}, {top}) - ({right}, {bottom})")
        print(f"Content size:      {bbox_width} x {bbox_height}")
        print(f"Content aspect:    {bbox_width / bbox_height:.4f}")
        print(f"Content width:     {bbox_width / width * 100:.2f}%")
        print(f"Content height:    {bbox_height / height * 100:.2f}%")
    else:
        print("Content bbox:      NONE")

print()
print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)
