"""
Data_Loader.py  —  Pascal VOC Few-Shot Segmentation (PROPER VERSION)
=====================================================================
This version correctly separates BASE classes (used for learning) from
NOVEL classes (used only at test time, with K support images).

Pascal VOC has 20 classes. Standard FSS split (fold 0):
  Base classes  (15): used in Phase 1 learning
  Novel classes  (5): NEVER seen in Phase 1, only shown at test time

We follow the standard 4-fold cross validation splits used in
PFENet, PANet, HSNet and other FSS papers.
"""

import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF

# ── Standard FSS splits for Pascal VOC ────────────────────────────
# Each fold holds 5 novel classes. The other 15 are base classes.
# We use fold 0 by default (same as most papers).
PASCAL_FSS_SPLITS = {
    0: [1,  2,  3,  4,  5],    # novel: aeroplane bicycle bird boat bottle
    1: [6,  7,  8,  9,  10],   # novel: bus car cat chair cow
    2: [11, 12, 13, 14, 15],   # novel: diningtable dog horse motorbike person
    3: [16, 17, 18, 19, 20],   # novel: pottedplant sheep sofa train tvmonitor
}

VOC_CLASS_NAMES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

IMG_SIZE = 224


def joint_transform(image, mask, augment=False):
    """Apply same spatial transform to image and mask together."""
    image = TF.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=Image.BILINEAR)
    mask  = TF.resize(mask,  (IMG_SIZE, IMG_SIZE), interpolation=Image.NEAREST)

    if augment and random.random() > 0.5:
        image = TF.hflip(image)
        mask  = TF.hflip(mask)

    image = TF.to_tensor(image)
    image = TF.normalize(image, mean=[0.485, 0.456, 0.406],
                                std= [0.229, 0.224, 0.225])
    mask  = torch.from_numpy(np.array(mask)).long()
    return image, mask


# ─────────────────────────────────────────────────────────────────
# Phase 1 Dataset — base classes, normal batch loading
# ─────────────────────────────────────────────────────────────────
class BaseClassDataset(Dataset):
    """
    Loads images containing BASE classes for Phase 1 learning.
    Returns: image, binary_mask, class_label (remapped 0..N_base-1)

    This is just like the original APM CIFAR loader — normal batches,
    no episodic sampling.
    """
    def __init__(self, voc_root, split, base_classes, augment=False):
        self.voc_root     = voc_root
        self.base_classes = base_classes
        self.augment      = augment
        self.label_map    = {c: i for i, c in enumerate(sorted(base_classes))}

        split_file = os.path.join(voc_root, "ImageSets", "Segmentation", split + ".txt")
        with open(split_file) as f:
            all_ids = [l.strip() for l in f if l.strip()]

        self.samples = []
        for img_id in all_ids:
            mask_path = os.path.join(voc_root, "SegmentationClass", img_id + ".png")
            if not os.path.exists(mask_path):
                continue
            mask = np.array(Image.open(mask_path))
            for cls_id in base_classes:
                if (mask == cls_id).any():
                    self.samples.append((img_id, cls_id))

        print(f"[BaseDataset] split={split} | {len(base_classes)} base classes "
              f"| {len(self.samples)} samples")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_id, cls_id = self.samples[idx]

        image = Image.open(
            os.path.join(self.voc_root, "JPEGImages", img_id + ".jpg")
        ).convert("RGB")
        mask = Image.open(
            os.path.join(self.voc_root, "SegmentationClass", img_id + ".png")
        )

        image, mask = joint_transform(image, mask, self.augment)

        # Binary mask: 1 = this class, 0 = background, 255 = ignore
        binary = torch.zeros_like(mask)
        binary[mask == 255]    = 255
        binary[mask == cls_id] = 1

        return image, binary, self.label_map[cls_id]


# ─────────────────────────────────────────────────────────────────
# Phase 2 & 3 Dataset — novel classes, support + query
# ─────────────────────────────────────────────────────────────────
class NovelClassDataset(Dataset):
    """
    Loads images for NOVEL classes used in Phase 2 (adaptation) and
    Phase 3 (testing).

    For each class, we keep all available images in a list.
    The FewShotTester will pick K of them as support and the rest as queries.
    """
    def __init__(self, voc_root, novel_classes):
        self.voc_root      = novel_classes
        self.novel_classes = novel_classes

        # Build a dict: class_id → list of image_ids containing that class
        self.class_images = {cls: [] for cls in novel_classes}

        # Use both train and val splits to get enough images
        all_ids = []
        for split in ["train", "val"]:
            split_file = os.path.join(voc_root, "ImageSets",
                                      "Segmentation", split + ".txt")
            if os.path.exists(split_file):
                with open(split_file) as f:
                    all_ids += [l.strip() for l in f if l.strip()]

        self.voc_root = voc_root
        for img_id in set(all_ids):
            mask_path = os.path.join(voc_root, "SegmentationClass", img_id + ".png")
            if not os.path.exists(mask_path):
                continue
            mask = np.array(Image.open(mask_path))
            for cls_id in novel_classes:
                if (mask == cls_id).any():
                    self.class_images[cls_id].append(img_id)

        for cls_id in novel_classes:
            n = len(self.class_images[cls_id])
            print(f"[NovelDataset] class={VOC_CLASS_NAMES[cls_id]:15s} "
                  f"(id={cls_id}) | {n} images available")

    def get_support_and_queries(self, cls_id, k_shot, seed=242):
        """
        For a given novel class, return:
          support_images : list of k_shot (image, binary_mask) tuples
          query_images   : list of remaining (image, binary_mask) tuples

        This is the KEY few-shot mechanism:
          - Support = the K examples the model is ALLOWED to see
          - Queries  = new images the model must segment without having
                       seen them before
        """
        random.seed(seed)
        imgs = self.class_images[cls_id].copy()

        if len(imgs) < k_shot + 1:
            raise ValueError(
                f"Class {VOC_CLASS_NAMES[cls_id]} only has {len(imgs)} images. "
                f"Need at least {k_shot + 1} (k_shot + 1 query)."
            )

        random.shuffle(imgs)
        support_ids = imgs[:k_shot]         # first K = support
        query_ids   = imgs[k_shot:]         # rest = queries

        support = [self._load(img_id, cls_id) for img_id in support_ids]
        queries = [self._load(img_id, cls_id) for img_id in query_ids]

        return support, queries

    def _load(self, img_id, cls_id):
        """Load one image and build its binary mask for cls_id."""
        image = Image.open(
            os.path.join(self.voc_root, "JPEGImages", img_id + ".jpg")
        ).convert("RGB")
        mask = Image.open(
            os.path.join(self.voc_root, "SegmentationClass", img_id + ".png")
        )

        image, mask = joint_transform(image, mask, augment=False)

        binary = torch.zeros_like(mask)
        binary[mask == 255]    = 255
        binary[mask == cls_id] = 1

        return image, binary


# ─────────────────────────────────────────────────────────────────
# Prepare base class loaders (for Phase 1)
# ─────────────────────────────────────────────────────────────────
def prepare_base_loaders(voc_root, fold=0, batch_size=8,
                         val_ratio=0.1, num_workers=2, seed=242):
    """
    Returns train/val DataLoaders for BASE classes (Phase 1 learning).
    """
    novel_classes = PASCAL_FSS_SPLITS[fold]
    base_classes  = [c for c in range(1, 21) if c not in novel_classes]

    full_ds = BaseClassDataset(voc_root, "train", base_classes, augment=True)
    eval_ds = BaseClassDataset(voc_root, "train", base_classes, augment=False)

    total   = len(full_ds)
    n_val   = int(total * val_ratio)
    n_train = total - n_val

    g = torch.Generator().manual_seed(seed)
    train_ds, _ = random_split(full_ds, [n_train, n_val], generator=g)
    _,  val_ds  = random_split(eval_ds, [n_train, n_val], generator=g)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"\n[Phase 1] Base classes: {[VOC_CLASS_NAMES[c] for c in base_classes]}")
    print(f"          Train={len(train_ds)} | Val={len(val_ds)}")

    return train_loader, val_loader, len(base_classes)


def prepare_novel_dataset(voc_root, fold=0):
    """Returns NovelClassDataset for Phase 2 & 3."""
    novel_classes = PASCAL_FSS_SPLITS[fold]
    return NovelClassDataset(voc_root, novel_classes), novel_classes
