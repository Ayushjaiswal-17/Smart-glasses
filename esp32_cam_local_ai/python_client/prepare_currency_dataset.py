"""Create deterministic train, validation, and test splits for INR notes."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image, ImageFile, ImageOps


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = ("train", "val", "test")
MAX_IMAGE_DIMENSION = 640

# Several source JPEGs are truncated but decodable. Normalizing them into the
# generated dataset keeps the original captures untouched during training.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def split_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    train = int(total * train_ratio)
    val = int(total * val_ratio)
    test = total - train - val
    if min(train, val, test) < 1:
        raise ValueError("Each class needs enough images for train, val, and test splits.")
    return train, val, test


def prepare_dataset(source: Path, destination: Path, seed: int, clean: bool = False) -> dict[str, dict[str, int]]:
    class_dirs = sorted(path for path in source.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No denomination folders found in {source}")

    if clean and destination.exists():
        shutil.rmtree(destination)

    summary: dict[str, dict[str, int]] = {}
    for class_dir in class_dirs:
        images = sorted(path for path in class_dir.iterdir() if path.suffix.lower() in VALID_EXTENSIONS)
        if not images:
            continue

        # A class-specific seed keeps each class independently reproducible.
        shuffled = images[:]
        random.Random(f"{seed}:{class_dir.name}").shuffle(shuffled)
        train_count, val_count, _ = split_counts(len(shuffled), 0.70, 0.15)
        assignments = {
            "train": shuffled[:train_count],
            "val": shuffled[train_count : train_count + val_count],
            "test": shuffled[train_count + val_count :],
        }

        summary[class_dir.name] = {}
        for split, paths in assignments.items():
            target = destination / split / class_dir.name
            target.mkdir(parents=True, exist_ok=True)
            for path in paths:
                target_path = target / f"{path.stem}.jpg"
                if target_path.exists():
                    continue
                with Image.open(path) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))
                    image.save(target_path, format="JPEG", quality=90)
            summary[class_dir.name][split] = len(paths)

    return summary


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Prepare INR denomination classifier dataset splits.")
    parser.add_argument("--source", type=Path, default=project_root / "Currency_images")
    parser.add_argument("--output", type=Path, default=project_root / "artifacts" / "inr_currency_cls")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true", help="Recreate all prepared images from scratch.")
    args = parser.parse_args()

    summary = prepare_dataset(args.source, args.output, args.seed, args.clean)
    print(f"Prepared dataset: {args.output}")
    for denomination, counts in summary.items():
        print(f"  INR {denomination}: train={counts['train']}, val={counts['val']}, test={counts['test']}")


if __name__ == "__main__":
    main()
