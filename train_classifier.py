"""
Train a SigLIP2 + Linear Head Brand Classifier on Real Datasets.

Uses:
  - dataset/our_logos: 52 protected brands with authentic background, crop, blur,
    and size variations (~1,100+ positive reference images).
  - dataset/competitor_logos: Real competitor logos (235 images across competitor,
    modified, and negative redesign categories) for the 'unknown' rejection class.

Usage:
    python train_classifier.py
    python train_classifier.py --epochs 35 --batch-size 64
"""

import sys
import argparse
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import torchvision.transforms as T
import torchvision.transforms.functional as TF

_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config


# ============================================================
# DATA AUGMENTATION
# ============================================================

class LogoAugmentor:
    """Generates augmented versions of logo images."""

    def __init__(self, target_size: int = 224):
        self.target_size = target_size

    def augment(self, img: Image.Image, n_augments: int = 3) -> List[Image.Image]:
        """Generate n augmented versions of the input image."""
        augmented = []
        # Always include the clean original resized & padded
        augmented.append(self._resize_pad(img))

        for _ in range(n_augments - 1):
            aug_img = img.copy()
            aug_img = self._apply_random_augmentation(aug_img)
            aug_img = self._resize_pad(aug_img)
            augmented.append(aug_img)

        return augmented

    def _resize_pad(self, img: Image.Image) -> Image.Image:
        """Resize and pad to target_size x target_size."""
        img = img.convert("RGB")
        img.thumbnail((self.target_size, self.target_size), Image.LANCZOS)
        padded = Image.new("RGB", (self.target_size, self.target_size), (0, 0, 0))
        x_off = (self.target_size - img.width) // 2
        y_off = (self.target_size - img.height) // 2
        padded.paste(img, (x_off, y_off))
        return padded

    def _apply_random_augmentation(self, img: Image.Image) -> Image.Image:
        """Apply a realistic combination of augmentations."""
        img = img.convert("RGB")

        # Slight rotation (-12 to +12 degrees)
        if random.random() < 0.4:
            angle = random.uniform(-12, 12)
            img = img.rotate(angle, expand=True, fillcolor=(0, 0, 0))

        # Horizontal flip (only 15% probability since text orientation matters)
        if random.random() < 0.15:
            img = ImageOps.mirror(img)

        # Scale variations (0.8x to 1.2x)
        if random.random() < 0.4:
            scale = random.uniform(0.8, 1.2)
            new_w = max(32, int(img.width * scale))
            new_h = max(32, int(img.height * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)

        # Random crop (80-100% of the image)
        if random.random() < 0.3:
            crop_frac = random.uniform(0.8, 1.0)
            cw = int(img.width * crop_frac)
            ch = int(img.height * crop_frac)
            x1 = random.randint(0, max(0, img.width - cw))
            y1 = random.randint(0, max(0, img.height - ch))
            img = img.crop((x1, y1, x1 + cw, y1 + ch))

        # Brightness
        if random.random() < 0.4:
            factor = random.uniform(0.75, 1.25)
            img = ImageEnhance.Brightness(img).enhance(factor)

        # Contrast
        if random.random() < 0.4:
            factor = random.uniform(0.8, 1.25)
            img = ImageEnhance.Contrast(img).enhance(factor)

        # Saturation
        if random.random() < 0.3:
            factor = random.uniform(0.7, 1.3)
            img = ImageEnhance.Color(img).enhance(factor)

        # Gaussian blur
        if random.random() < 0.2:
            radius = random.uniform(0.4, 1.5)
            img = img.filter(ImageFilter.GaussianBlur(radius))

        # Sharpness
        if random.random() < 0.2:
            factor = random.uniform(1.2, 2.5)
            img = ImageEnhance.Sharpness(img).enhance(factor)

        # Background color composite
        if random.random() < 0.25:
            bg_choice = random.choice([
                (0, 0, 0), (255, 255, 255), (30, 30, 30),
                (128, 128, 128), (15, 23, 42)
            ])
            bg = Image.new("RGB", img.size, bg_choice)
            if img.mode == "RGBA":
                bg.paste(img, mask=img.split()[3])
            else:
                bg.paste(img)
            img = bg

        return img


# ============================================================
# DATASET LOADING
# ============================================================

def load_brand_images(logos_dir: Path, brand_db_path: Path) -> Dict[str, List[Image.Image]]:
    """
    Load all genuine reference images per brand from BrandDatabase.
    Includes primary logo, all positive dataset variants, and favicons.
    """
    from src.database.brand_database import BrandDatabase
    db = BrandDatabase()
    brand_images = {}

    for brand in db.brand_names:
        paths = db.brand_to_ref_paths.get(brand, [])
        imgs = []
        for p in paths:
            if not p.exists():
                continue
            try:
                img = Image.open(p)
                if getattr(img, "is_animated", False):
                    img.seek(0)
                imgs.append(img.convert("RGB"))
            except Exception as e:
                pass

        if not imgs:
            fname = db.brand_to_filename.get(brand, f"{brand}.png")
            alt = logos_dir / fname
            if alt.exists():
                try:
                    imgs.append(Image.open(alt).convert("RGB"))
                except Exception:
                    pass

        brand_images[brand] = imgs

    return brand_images


def load_competitor_images(competitor_dir: Path) -> List[Image.Image]:
    """
    Load real competitor logos, modified logos, and redesigns from dataset/competitor_logos.
    """
    if not competitor_dir.exists():
        print(f"  [WARN] Competitor directory not found: {competitor_dir}")
        return []

    imgs = []
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".avif", ".bmp"}

    for p in competitor_dir.rglob("*.*"):
        if p.is_file() and p.suffix.lower() in valid_exts:
            try:
                img = Image.open(p)
                if getattr(img, "is_animated", False):
                    img.seek(0)
                imgs.append(img.convert("RGB"))
            except Exception:
                pass

    print(f"  [OK] Loaded {len(imgs)} competitor & negative images from {competitor_dir}")
    return imgs


def generate_unknown_samples(
    brand_images: Dict[str, List[Image.Image]],
    competitor_images: List[Image.Image],
    n_samples: int = 500,
    target_size: int = 224,
    augmentor: Optional[LogoAugmentor] = None,
) -> List[Image.Image]:
    """
    Generate unknown class samples combining:
      1. Real competitor logos and negative redesigns (with augmentations)
      2. Mixed brand combinations and patch shuffling
      3. Random synthetic textures, noise, and gradients
    """
    unknown_imgs = []
    augmentor = augmentor or LogoAugmentor(target_size=target_size)

    # 1. Incorporate real competitor images
    if competitor_images:
        for c_img in competitor_images:
            unknown_imgs.append(augmentor._resize_pad(c_img))

        # Augment competitor images to increase coverage
        needed_from_competitors = max(0, min(len(competitor_images) * 2, n_samples - len(unknown_imgs)))
        for _ in range(needed_from_competitors):
            c_src = random.choice(competitor_images)
            aug_c = augmentor._apply_random_augmentation(c_src.copy())
            unknown_imgs.append(augmentor._resize_pad(aug_c))

    # 2. Add synthetic hard negatives to guarantee generalization to non-logo images
    flat_brands = [imgs[0] for imgs in brand_images.values() if imgs]
    remaining = max(50, n_samples - len(unknown_imgs))

    for _ in range(remaining):
        strategy = random.choice(["noise", "solid", "gradient", "patch_shuffle", "mixed_brands"])

        if strategy == "noise":
            arr = np.random.randint(0, 256, (target_size, target_size, 3), dtype=np.uint8)
            img = Image.fromarray(arr)

        elif strategy == "solid":
            color = tuple(random.randint(0, 255) for _ in range(3))
            img = Image.new("RGB", (target_size, target_size), color)

        elif strategy == "gradient":
            arr = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            for c in range(3):
                start_val = random.randint(0, 255)
                end_val = random.randint(0, 255)
                gradient = np.linspace(start_val, end_val, target_size)
                if random.random() < 0.5:
                    arr[:, :, c] = gradient[np.newaxis, :]
                else:
                    arr[:, :, c] = gradient[:, np.newaxis]
            img = Image.fromarray(arr)

        elif strategy == "patch_shuffle" and flat_brands:
            src = random.choice(flat_brands).copy().convert("RGB")
            src = src.resize((target_size, target_size), Image.LANCZOS)
            arr = np.array(src)
            patch_h, patch_w = target_size // 4, target_size // 4
            patches = []
            for r in range(0, target_size, patch_h):
                for c in range(0, target_size, patch_w):
                    patches.append(arr[r:r+patch_h, c:c+patch_w].copy())
            random.shuffle(patches)
            idx = 0
            for r in range(0, target_size, patch_h):
                for c in range(0, target_size, patch_w):
                    if idx < len(patches):
                        h, w = patches[idx].shape[:2]
                        arr[r:r+h, c:c+w] = patches[idx]
                        idx += 1
            arr = arr.astype(np.float32)
            for c_idx in range(3):
                arr[:, :, c_idx] = np.clip(arr[:, :, c_idx] + random.uniform(-60, 60), 0, 255)
            img = Image.fromarray(arr.astype(np.uint8))

        elif strategy == "mixed_brands" and len(flat_brands) >= 2:
            src1, src2 = random.sample(flat_brands, 2)
            src1 = src1.copy().convert("RGB").resize((target_size, target_size), Image.LANCZOS)
            src2 = src2.copy().convert("RGB").resize((target_size, target_size), Image.LANCZOS)
            alpha = random.uniform(0.3, 0.7)
            arr = (np.array(src1).astype(np.float32) * alpha +
                   np.array(src2).astype(np.float32) * (1 - alpha))
            img = Image.fromarray(arr.astype(np.uint8))

        else:
            img = Image.new("RGB", (target_size, target_size), (128, 128, 128))

        unknown_imgs.append(img)

    return unknown_imgs


# ============================================================
# CLASSIFIER MODEL
# ============================================================

class BrandClassifier(nn.Module):
    """
    MLP classifier on top of frozen SigLIP2 embeddings.
    Architecture: 768 -> 256 -> 128 -> num_classes
    """
    def __init__(self, embedding_dim: int = 768, num_classes: int = 53):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)

    def predict_proba(self, x):
        """Return softmax probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)


# ============================================================
# TRAINING WORKFLOW
# ============================================================

def train_classifier(
    our_logos_dir: Path = config.OUR_LOGOS_DIR,
    competitor_dir: Path = config.COMPETITOR_LOGOS_DIR,
    augments_per_brand: int = 60,
    unknown_samples: int = 500,
    epochs: int = 35,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    val_split: float = 0.15,
):
    print("=" * 68)
    print("BRAND CLASSIFIER TRAINING (SIGLIP 2 + REAL DATASETS)")
    print("=" * 68)
    print(f"Our logos dir       : {our_logos_dir}")
    print(f"Competitor logos dir: {competitor_dir}")

    # 1. Load brand images
    print("\n[1/5] Loading authentic brand images & variations...")
    brand_images = load_brand_images(config.LOGOS_DIR, config.BRAND_DB_PATH)
    brand_names = sorted(brand_images.keys())
    num_brands = len(brand_names)
    total_loaded_refs = sum(len(imgs) for imgs in brand_images.values())
    print(f"  Loaded {total_loaded_refs} real reference images across {num_brands} brands (avg {total_loaded_refs/num_brands:.1f}/brand)")

    # 2. Load competitor logos
    print("\n[2/5] Loading competitor logos...")
    competitor_images = load_competitor_images(competitor_dir)

    # Class mapping: 0..51 = brands, 52 = unknown
    class_names = brand_names + ["unknown"]
    num_classes = len(class_names)
    brand_to_idx = {b: i for i, b in enumerate(class_names)}
    unknown_idx = brand_to_idx["unknown"]

    # 3. Generate augmented training data
    print(f"\n[3/5] Generating dataset samples (~{augments_per_brand} per brand)...")
    target_size = getattr(config, "SIGLIP_INPUT_SIZE", 384)
    augmentor = LogoAugmentor(target_size=target_size)

    all_images = []
    all_labels = []

    for brand in brand_names:
        imgs = brand_images[brand]
        if not imgs:
            continue
        aug_per_img = max(2, augments_per_brand // len(imgs))
        brand_aug = []
        for src_img in imgs:
            brand_aug.extend(augmentor.augment(src_img, n_augments=aug_per_img))
        all_images.extend(brand_aug)
        all_labels.extend([brand_to_idx[brand]] * len(brand_aug))

    print(f"  Generated {len(all_images)} positive brand samples.")

    # Generate unknown class samples
    print(f"  Generating {unknown_samples} 'unknown' class samples (real competitors + negatives)...")
    unknown_imgs = generate_unknown_samples(
        brand_images=brand_images,
        competitor_images=competitor_images,
        n_samples=unknown_samples,
        target_size=target_size,
        augmentor=augmentor,
    )
    all_images.extend(unknown_imgs)
    all_labels.extend([unknown_idx] * len(unknown_imgs))

    total_samples = len(all_images)
    print(f"  Total combined training samples: {total_samples} ({num_brands} brands + 1 unknown class)")

    # 4. Extract SigLIP2 embeddings
    cache_embs_path = config.SIGLIP_VERSION_DIR / "cached_train_embeddings.pt"
    if cache_embs_path.exists():
        print(f"\n[4/5] Loading cached SigLIP2 embeddings from {cache_embs_path}...")
        cached_data = torch.load(cache_embs_path, map_location="cpu", weights_only=False)
        embeddings = cached_data["embeddings"]
        labels = cached_data["labels"]
        embedding_dim = embeddings.shape[1]
        print(f"  Loaded {len(embeddings)} embeddings (dim: {embedding_dim})")
    else:
        print("\n[4/5] Extracting SigLIP2 embeddings on GPU...")
        from models.siglip_model import SigLIPModel

        siglip = SigLIPModel()
        print(f"  SigLIP Model: {siglip.model_name} on {siglip.device} ({siglip.dtype})")

        all_embeddings = []
        emb_batch_size = 32

        for i in range(0, len(all_images), emb_batch_size):
            batch = all_images[i:i + emb_batch_size]
            inputs = siglip.processor(images=batch, return_tensors="pt").to(siglip.device)
            if siglip.dtype == torch.float16:
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                        inputs[k] = v.half()

            with torch.inference_mode():
                outputs = siglip.model.vision_model(**inputs)
                emb = outputs.pooler_output
                emb = emb / (emb.norm(p=2, dim=-1, keepdim=True) + 1e-12)
                all_embeddings.append(emb.cpu())

            if (i // emb_batch_size) % 25 == 0 or (i + emb_batch_size) >= len(all_images):
                processed = min(i + emb_batch_size, len(all_images))
                print(f"  Embedded {processed:4d}/{len(all_images)} images ({processed/len(all_images)*100:5.1f}%)...")

        embeddings = torch.cat(all_embeddings, dim=0)
        labels = torch.tensor(all_labels, dtype=torch.long)
        embedding_dim = embeddings.shape[1]

        # Cache embeddings to disk
        config.SIGLIP_VERSION_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"embeddings": embeddings, "labels": labels}, cache_embs_path)

        # Free vision model from VRAM
        del siglip
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 5. Train/Val split & Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[5/5] Training BrandClassifier for {epochs} epochs on {device}...")
    embeddings = embeddings.float()
    indices = torch.randperm(len(embeddings))
    embeddings = embeddings[indices]
    labels = labels[indices]

    val_size = int(len(embeddings) * val_split)
    train_emb, val_emb = embeddings[val_size:], embeddings[:val_size]
    train_lbl, val_lbl = labels[val_size:], labels[:val_size]

    train_ds = TensorDataset(train_emb, train_lbl)
    val_ds = TensorDataset(val_emb, val_lbl)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    classifier = BrandClassifier(embedding_dim=embedding_dim, num_classes=num_classes)
    classifier = classifier.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(classifier.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_emb, batch_lbl in train_loader:
            batch_emb = batch_emb.to(device)
            batch_lbl = batch_lbl.to(device)

            optimizer.zero_grad()
            logits = classifier(batch_emb)
            loss = criterion(logits, batch_lbl)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_lbl)
            train_correct += (logits.argmax(dim=-1) == batch_lbl).sum().item()
            train_total += len(batch_lbl)

        scheduler.step()

        # Validation
        classifier.eval()
        val_correct = 0
        val_total = 0
        unknown_correct = 0
        unknown_total = 0

        with torch.no_grad():
            for batch_emb, batch_lbl in val_loader:
                batch_emb = batch_emb.to(device)
                batch_lbl = batch_lbl.to(device)
                logits = classifier(batch_emb)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == batch_lbl).sum().item()
                val_total += len(batch_lbl)

                unk_mask = (batch_lbl == unknown_idx)
                if unk_mask.sum() > 0:
                    unknown_correct += (preds[unk_mask] == unknown_idx).sum().item()
                    unknown_total += unk_mask.sum().item()

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        unk_acc = unknown_correct / max(unknown_total, 1)
        avg_loss = train_loss / max(train_total, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = classifier.state_dict().copy()

        if (epoch + 1) % 5 == 0 or epoch == 0 or (epoch + 1) == epochs:
            print(f"  Epoch {epoch+1:2d}/{epochs:2d}: loss={avg_loss:.4f} "
                  f"train_acc={train_acc*100:5.2f}% val_acc={val_acc*100:5.2f}% "
                  f"unknown_rejection={unk_acc*100:5.2f}% "
                  f"{'*BEST*' if val_acc >= best_val_acc else ''}")

    if best_state:
        classifier.load_state_dict(best_state)

    # Save trained checkpoint (both versioned and default paths)
    save_path = config.REFERENCE_DIR / "brand_classifier.pt"
    versioned_dir = getattr(config, "SIGLIP_VERSION_DIR", config.REFERENCE_DIR)
    versioned_dir.mkdir(parents=True, exist_ok=True)
    versioned_save_path = versioned_dir / "brand_classifier.pt"

    checkpoint_payload = {
        "model_state_dict": classifier.cpu().state_dict(),
        "class_names": class_names,
        "brand_names": brand_names,
        "num_classes": num_classes,
        "embedding_dim": embedding_dim,
        "model_id": config.SIGLIP_MODEL_NAME,
        "best_val_accuracy": best_val_acc,
        "augments_per_brand": augments_per_brand,
        "unknown_samples": unknown_samples,
        "epochs": epochs,
    }
    torch.save(checkpoint_payload, save_path)
    torch.save(checkpoint_payload, versioned_save_path)

    print("\n" + "=" * 68)
    print("TRAINING FINISHED & SAVED")
    print("=" * 68)
    print(f"Model saved to        : {save_path}")
    print(f"Total classes trained : {num_classes} ({num_brands} brands + 1 unknown class)")
    print(f"Best validation acc   : {best_val_acc * 100:.2f}%")
    print("=" * 68)

    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Brand Classifier on Real Datasets")
    parser.add_argument("--augments-per-brand", type=int, default=60,
                        help="Number of augmented samples per brand (default: 60)")
    parser.add_argument("--unknown-samples", type=int, default=500,
                        help="Number of 'unknown' class samples (default: 500)")
    parser.add_argument("--epochs", type=int, default=35,
                        help="Training epochs (default: 35)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64)")

    args = parser.parse_args()

    train_classifier(
        augments_per_brand=args.augments_per_brand,
        unknown_samples=args.unknown_samples,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )
