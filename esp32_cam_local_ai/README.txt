Smart Glasses for Blind Assistance - Local Test

Hardware
  - ESP32-CAM for image capture
  - ESP8266 for sensor communication (future)
  - ToF sensor + MPU6050 (future)

ESP32 Firmware
  1. Open esp32cam_firmware/esp32cam_webserver.ino in Arduino IDE or PlatformIO.
  2. Copy secrets.example.h to secrets.h and set Wi-Fi credentials there.
  3. Flash to ESP32-CAM.
  4. Open Serial Monitor at 115200 to read the camera IP.
  5. Verify /jpg and /stream in a browser. /stream is the live MJPEG feed.

Python Client
  1. Install dependencies:
       pip install -r python_client/requirements.txt
  2. Place trained YOLO weights at:
       python_client/models/best.pt
  3. Run low-latency stream inference (default):
       python python_client/detect.py --ip <camera-ip>

Model Scope
  Detects: person, chair, table, wall, door, stair, vehicle, bicycle.
  Does not include face detection.

Notes
  - Start with snapshot mode if /stream is unstable.
  - Use good lighting for camera testing.
  - Keep obstacle sensing local in production for safety.
