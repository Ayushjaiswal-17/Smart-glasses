import argparse
import os
import cv2
import time
import pyttsx3
import numpy as np
import requests
import queue
import threading
from collections import deque

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False

    class YOLO:
        """Mock YOLO model if ultralytics isn't installed."""

        def __init__(self, model):
            self.model = model

        def __call__(self, frame, verbose=False):
            class Box:
                def __init__(self, cls, conf):
                    self.cls = [cls]
                    self.conf = [conf]

            class Result:
                def __init__(self, boxes, names):
                    self.boxes = boxes
                    self.names = names

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            if cv2.countNonZero(edges) > 12000:
                return [Result([Box(0, 0.75)], {0: "person"})]

            return [Result([], {})]


DEFAULT_IP = "192.168.156.251"

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "best.pt"
)

FALLBACK_MODEL = "yolov8n.pt"

CURRENCY_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "currency_classifier.pt",
)

LABELS_OF_INTEREST = {
    "person",
    "chair",
    "table",
    "wall",
    "door",
    "stair",
    "vehicle",
    "bicycle",
}

CURRENCY_NAMES = {
    "100": "one hundred rupees",
    "200": "two hundred rupees",
    "500": "five hundred rupees",
}


_speech_queue = queue.Queue(maxsize=4)


def _speech_worker():
    """Keep audio playback off the camera/inference loop."""
    try:
        engine = pyttsx3.init()
        while True:
            message = _speech_queue.get()
            try:
                if message is None:
                    return
                engine.say(message)
                engine.runAndWait()
            except Exception as error:
                print("TTS Error:", error)
            finally:
                _speech_queue.task_done()
    except Exception as error:
        print("TTS Error:", error)


_speech_thread = threading.Thread(target=_speech_worker, daemon=True)
_speech_thread.start()


def speak(text):
    # Keep recent alerts flowing if speech playback falls behind.
    try:
        _speech_queue.put_nowait(text)
    except queue.Full:
        try:
            _speech_queue.get_nowait()
            _speech_queue.task_done()
        except queue.Empty:
            pass
        _speech_queue.put_nowait(text)


class CurrencySmoother:
    """Announce a denomination only after several confident frame predictions."""

    def __init__(self, window_size=3, minimum_matches=2, confidence=0.80):
        self.predictions = deque(maxlen=window_size)
        self.minimum_matches = minimum_matches
        self.confidence = confidence
        self.last_announcement = ""
        self.last_announcement_time = 0

    def update(self, results):
        if not results or getattr(results[0], "probs", None) is None:
            return None

        result = results[0]
        label = str(result.names[result.probs.top1])
        confidence = float(result.probs.top1conf)
        if label not in CURRENCY_NAMES or confidence < self.confidence:
            self.predictions.clear()
            return None

        self.predictions.append(label)
        if len(self.predictions) < self.minimum_matches:
            return None

        matches = sum(item == label for item in self.predictions)
        now = time.time()
        if matches >= self.minimum_matches and (
            label != self.last_announcement or now - self.last_announcement_time > 8
        ):
            self.last_announcement = label
            self.last_announcement_time = now
            return CURRENCY_NAMES[label], confidence

        return None


class FeedbackState:
    def __init__(self, object_every=2):
        self.last_announcement = ""
        self.last_detection_time = 0
        self.currency_smoother = CurrencySmoother()
        self.object_every = max(1, object_every)
        self.frame_count = 0


def process_frame(frame, object_model, currency_model, state):
    state.frame_count += 1
    # Currency is the latency-sensitive result, so run it before general object detection.
    current_time = time.time()
    if currency_model is not None:
        currency_result = state.currency_smoother.update(currency_model(frame, verbose=False))
        if currency_result:
            denomination, confidence = currency_result
            print(f"Currency: {denomination} ({confidence:.0%})")
            speak(denomination)

    detected = []
    annotated = frame
    if state.frame_count % state.object_every == 1 or state.object_every == 1:
        results = object_model(frame, verbose=False)
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = result.names[cls]

                if label in LABELS_OF_INTEREST and conf > 0.5:
                    detected.append((label, conf))

        if ULTRALYTICS_AVAILABLE:
            annotated = results[0].plot()

    if detected:
        detected.sort(key=lambda x: x[1], reverse=True)

        names = list(dict.fromkeys([d[0] for d in detected]))
        announcement = ", ".join(names)

        cv2.putText(
            annotated,
            announcement,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        if (
            announcement != state.last_announcement
            or current_time - state.last_detection_time > 5
        ):
            print("Detected:", announcement)
            speak(announcement)
            state.last_announcement = announcement
            state.last_detection_time = current_time

    cv2.imshow("ESP32-CAM YOLO Detection", annotated)
    return cv2.waitKey(1) & 0xFF != ord("q")


def run_stream_mode(base_url, object_model, currency_model=None, object_every=2):
    stream_url = f"{base_url}/stream"
    print(f"Connecting to {stream_url}")
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("OpenCV could not open the MJPEG stream.")
        return False

    # Continuously drain the stream, keeping just one frame so inference always
    # works on the newest camera position instead of processing a stale backlog.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    latest_frame = queue.Queue(maxsize=1)
    stop_reader = threading.Event()

    def read_stream():
        while not stop_reader.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            try:
                latest_frame.put_nowait(frame)
            except queue.Full:
                try:
                    latest_frame.get_nowait()
                except queue.Empty:
                    pass
                try:
                    latest_frame.put_nowait(frame)
                except queue.Full:
                    pass

    reader = threading.Thread(target=read_stream, daemon=True)
    reader.start()
    state = FeedbackState(object_every)
    failed_reads = 0
    try:
        while True:
            try:
                frame = latest_frame.get(timeout=1)
            except queue.Empty:
                failed_reads += 1
                if failed_reads >= 5:
                    print("MJPEG stream stopped responding.")
                    return False
                continue

            failed_reads = 0
            if not process_frame(frame, object_model, currency_model, state):
                return True
    finally:
        stop_reader.set()
        cap.release()
        reader.join(timeout=1)
        cv2.destroyAllWindows()


def run_snapshot_mode(base_url, object_model, currency_model=None, object_every=2):
    snapshot_url = f"{base_url}/jpg"
    print(f"Using snapshot fallback: {snapshot_url}")
    state = FeedbackState(object_every)

    while True:
        try:
            response = requests.get(snapshot_url, timeout=5)
            response.raise_for_status()
            frame = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Camera returned an invalid JPEG frame")
        except (requests.RequestException, ValueError) as error:
            print(f"Snapshot request failed: {error}")
            time.sleep(1)
            continue

        if not process_frame(frame, object_model, currency_model, state):
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="ESP32-CAM Stream YOLO Client"
    )

    parser.add_argument(
        "--ip",
        default=DEFAULT_IP,
        help="ESP32-CAM IP Address",
    )

    parser.add_argument(
        "--currency-model",
        default=CURRENCY_MODEL_PATH,
        help="Path to the trained INR denomination classification model.",
    )

    parser.add_argument(
        "--disable-currency",
        action="store_true",
        help="Run object detection only.",
    )

    parser.add_argument(
        "--mode",
        choices=("snapshot", "stream"),
        default="stream",
        help="Camera transport. Stream mode stays current when inference is slower than capture.",
    )

    parser.add_argument(
        "--object-every",
        type=int,
        default=2,
        help="Run general object detection once every N frames; currency is checked on every frame.",
    )

    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help="YOLO model path",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.model):
        if args.model == MODEL_PATH:
            print("Using fallback model:", FALLBACK_MODEL)
            args.model = FALLBACK_MODEL
        else:
            print("Model not found:", args.model)
            return

    print("Loading model...")
    object_model = YOLO(args.model)

    currency_model = None
    if not args.disable_currency:
        if os.path.isfile(args.currency_model):
            print("Loading currency classifier...")
            currency_model = YOLO(args.currency_model)
        else:
            print("Currency classifier not found; continuing without currency recognition.")
            print("Train it with: python train_currency_classifier.py")

    base_url = f"http://{args.ip}"

    try:
        if args.mode == "snapshot":
            run_snapshot_mode(base_url, object_model, currency_model, args.object_every)
        else:
            stream_finished_normally = run_stream_mode(base_url, object_model, currency_model, args.object_every)
            if not stream_finished_normally:
                run_snapshot_mode(base_url, object_model, currency_model, args.object_every)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
