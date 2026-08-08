# Smart Glasses for Blind Assistance

Prototype for blind and visually impaired users. ESP32-CAM streams video to a local laptop running YOLO object detection. Feedback via TTS audio, optional vibration.

## Current Setup

- ESP32-CAM serves individual frames at `/jpg` and a live MJPEG feed at `/stream`.
- Local Python client consumes the live feed and runs YOLO inference on the newest frame.
- Detected objects: person, chair, table, wall, door, stair, vehicle, bicycle.
- Face detection intentionally excluded from scope.

## Hardware

- ESP32-CAM: camera + Wi-Fi stream server.
- ESP8266: reserved for sensor telemetry (not yet integrated).
- ToF sensor: reserved for obstacle distance (not yet integrated).
- MPU6050 IMU: reserved for motion/tilt (not yet integrated).

## Project Structure

```text
esp32_cam_local_ai/
├── esp32cam_firmware/
│   └── esp32cam_webserver.ino
├── python_client/
│   ├── detect.py
│   ├── requirements.txt
│   └── models/
└── README.txt
```

## What Works Now

- ESP32-CAM web server with `/jpg` snapshot endpoint.
- Python snapshot-mode detector using OpenCV, YOLO, and pyttsx3.
- Placeholder model directory ready for weights.

## INR Currency Recognition

The project includes a trained lightweight classifier for the available INR `100`,
`200`, and `500` denomination images. It recognizes one clearly visible note at a
time; it does not verify authenticity or count multiple notes.

Train it again after adding images:

```bash
pip install -r esp32_cam_local_ai/python_client/requirements.txt
python esp32_cam_local_ai/python_client/train_currency_classifier.py
```

The script creates deterministic `train`, `val`, and `test` splits in `artifacts/`,
evaluates the held-out test split, and writes the runtime model to
`esp32_cam_local_ai/python_client/models/currency_classifier.pt`.

For useful real-world accuracy, capture ESP32-CAM images before retraining. Collect
at least 150 varied samples for each denomination and 300 samples with no note:

```bash
python esp32_cam_local_ai/python_client/capture_currency_samples.py --ip <esp32-cam-ip> --label 100
python esp32_cam_local_ai/python_client/capture_currency_samples.py --ip <esp32-cam-ip> --label 200
python esp32_cam_local_ai/python_client/capture_currency_samples.py --ip <esp32-cam-ip> --label 500
python esp32_cam_local_ai/python_client/capture_currency_samples.py --ip <esp32-cam-ip> --label not_currency
```

Use multiple backgrounds, distances, orientations, lighting levels, note sides, and
partly folded notes. For `not_currency`, capture normal scenes, hands, tables, and
other objects with no currency. Then rerun the training command above.

Run the camera client with currency recognition enabled by default:

```bash
python esp32_cam_local_ai/python_client/detect.py --ip <esp32-cam-ip>
```

The client uses `/stream` by default and keeps only the newest frame when
inference falls behind, reducing delay while the camera moves. Use
`--mode snapshot` if your network cannot keep the MJPEG stream stable, or
`--disable-currency` to run object detection alone. The stream occupies the
camera's single web-server connection while it is active.

## Next Steps

1. Add or train YOLO weights and copy to `python_client/models/best.pt`.
2. Copy `esp32_cam_local_ai/esp32cam_firmware/secrets.example.h` to
   `secrets.h`, set your Wi-Fi credentials there, then flash the firmware.
3. Install Python dependencies:
   ```bash
   pip install -r esp32_cam_local_ai/python_client/requirements.txt
   ```
4. Run local inference:
   ```bash
   python esp32_cam_local_ai/python_client/detect.py --ip <esp32-cam-ip>
   ```
5. Verify `/jpg` and `/stream` in a browser first. `/stream` is the live feed.
6. Integrate ESP8266 + ToF distance + MPU6050 motion data.
7. Move obstacle sensing fully local for safety.
8. Add AWS/cloud backend only after local testing is stable.
