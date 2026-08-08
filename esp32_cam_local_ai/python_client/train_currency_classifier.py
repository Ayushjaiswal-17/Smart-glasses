"""Train and evaluate a lightweight INR denomination classifier."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from prepare_currency_dataset import prepare_dataset


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Train an INR currency classifier with Ultralytics YOLO.")
    parser.add_argument("--data", type=Path, default=project_root / "artifacts" / "inr_currency_cls")
    parser.add_argument("--source", type=Path, default=project_root / "Currency_images")
    parser.add_argument("--model", default="yolo11n-cls.pt", help="Base Ultralytics classification model.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=-1, help="-1 lets Ultralytics select a safe batch size.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=project_root / "runs" / "currency_classifier")
    args = parser.parse_args()

    prepare_dataset(args.source, args.data, args.seed)
    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name="train",
        exist_ok=True,
        seed=args.seed,
        deterministic=True,
        patience=12,
        degrees=180,
        fliplr=0.5,
        flipud=0.5,
    )

    best_weights = args.project / "train" / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"Training did not produce expected weights: {best_weights}")

    best_model = YOLO(str(best_weights))
    metrics = best_model.val(data=str(args.data), split="test", imgsz=args.imgsz, device=args.device)
    print("Test metrics:", metrics.results_dict)

    model_output = project_root / "esp32_cam_local_ai" / "python_client" / "models" / "currency_classifier.pt"
    model_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, model_output)
    print(f"Currency classifier saved to: {model_output}")


if __name__ == "__main__":
    main()
