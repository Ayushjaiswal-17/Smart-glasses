"""Capture labeled ESP32-CAM images for INR currency classifier training."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests


VALID_LABELS = ("100", "200", "500", "not_currency")


def fetch_frame(url: str) -> np.ndarray | None:
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Capture labeled ESP32-CAM currency images.")
    parser.add_argument("--ip", required=True, help="ESP32-CAM IP address")
    parser.add_argument("--label", required=True, choices=VALID_LABELS)
    parser.add_argument("--output", type=Path, default=project_root / "Currency_images")
    args = parser.parse_args()

    destination = args.output / args.label
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_url = f"http://{args.ip}/jpg"
    saved = 0

    print("Press S to save the current frame, Q to quit.")
    while True:
        try:
            frame = fetch_frame(snapshot_url)
            if frame is None:
                raise ValueError("Camera returned an invalid JPEG")
        except (requests.RequestException, ValueError) as error:
            print(f"Camera request failed: {error}")
            time.sleep(1)
            continue

        preview = frame.copy()
        cv2.putText(preview, f"Label: {args.label} | S: save | Q: quit", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Capture INR currency samples", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("s"):
            filename = datetime.now().strftime("esp32_%Y%m%d_%H%M%S_%f.jpg")
            if cv2.imwrite(str(destination / filename), frame):
                saved += 1
                print(f"Saved {destination / filename}")

    cv2.destroyAllWindows()
    print(f"Saved {saved} images for label '{args.label}'.")


if __name__ == "__main__":
    main()
