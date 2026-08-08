# What To Do Next

## 1. Obtain the YOLO model

- Train a YOLOv8/v9 model on a dataset containing:
  - person, chair, table, wall, door, stair, vehicle, bicycle
  - optional: currency notes
- Alternatively download a pretrained blind-assistance/object-detection model.
- Save weights as `esp32_cam_local_ai/python_client/models/best.pt`.

## 2. Flash the firmware

- Open `esp32_cam_local_ai/esp32cam_firmware/esp32cam_webserver.ino`.
- Copy `secrets.example.h` to `secrets.h` and set the target Wi-Fi credentials there.
- Select board AI Thinker ESP32-CAM.
- Flash and open Serial Monitor at 115200.
- Note the IP address printed after `WiFi Connected`.

## 3. Verify camera endpoints

- Visit `http://<camera-ip>/jpg` in a browser. A single frame should appear.
- Visit `http://<camera-ip>/stream` in a browser. MJPEG stream should play.

## 4. Install Python dependencies

```bash
cd esp32_cam_local_ai/python_client
pip install -r requirements.txt
```

## 5. Run the detector

```bash
python detect.py --ip <camera-ip>
```

The script will announce detected objects via TTS.

## 6. Tune detection behavior

- Adjust label set in `detect.py` -> `LABELS_OF_INTEREST`.
- Adjust confidence threshold if too many or too few detections.
- Increase `--object-every` on a slower laptop to reserve more processing time for currency checks.

## 7. Live stream mode

- The firmware serves an MJPEG feed at `/stream`; the client uses it by default.
- The client drops queued old frames to keep inference close to the camera's current position.
- Use `python detect.py --ip <camera-ip> --mode snapshot` if the stream is unstable.

## 8. Integrate ESP8266 sensors

- Attach ESP8266 to ToF sensor and MPU6050.
- Send distance/IMU data to laptop over Wi-Fi (HTTP or MQTT).
- Fuse sensor data with object labels in `detect.py`.

## 9. Local obstacle safety

- Keep distance measurement and critical obstacle alerting on-device or local.
- Do not rely only on cloud path for safety-critical warnings.

## 10. Cloud/AWS phase

- Only after local pipeline is stable and tested.
- Consider AWS IoT Core + Lambda + S3 for model hosting if needed.

## 11. Hardware enclosure

- Design glasses mount for ESP32-CAM, bone-conduction headset or earbuds, and vibration motor.
- Ensure camera field of view matches user's forward gaze.
