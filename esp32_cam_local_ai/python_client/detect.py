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
    """Keep audio playback off inference and recover if the TTS engine fails."""
    engine = None
    while True:
        message = _speech_queue.get()
        try:
            if message is None:
                return

            if engine is None:
                engine = pyttsx3.init()

            engine.say(message)
            engine.runAndWait()
        except Exception as error:
            print("TTS Error:", error)
            engine = None
            time.sleep(1)
        finally:
            _speech_queue.task_done()


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
    """Announce each stable note detection once, then re-arm after it leaves view."""

    def __init__(
        self,
        window_size=3,
        minimum_matches=2,
        confidence=0.80,
        absence_frames=3,
        repeat_interval=3.0,
    ):
        self.predictions = deque(maxlen=window_size)
        self.minimum_matches = minimum_matches
        self.confidence = confidence
        self.absence_frames = absence_frames
        self.repeat_interval = repeat_interval
        self.absent_count = 0
        self.last_announcement = ""
        self.last_announcement_time = 0.0

    @property
    def note_visible(self):
        # A confident first frame is enough to avoid classifying Gandhi's
        # portrait as a person while the currency decision is being confirmed.
        return bool(self.last_announcement or self.predictions)

    def update(self, results):
        result = results[0] if results else None
        probs = getattr(result, "probs", None) if result is not None else None
        if probs is None:
            label, confidence = None, 0.0
        else:
            label = str(result.names[probs.top1])
            confidence = float(probs.top1conf)

        if label not in CURRENCY_NAMES or confidence < self.confidence:
            self.predictions.clear()
            self.absent_count += 1
            if self.absent_count >= self.absence_frames:
                self.last_announcement = ""
                self.last_announcement_time = 0.0
            return None

        self.absent_count = 0
        self.predictions.append(label)
        if len(self.predictions) < self.minimum_matches:
            return None

        matches = sum(item == label for item in self.predictions)
        now = time.monotonic()
        if matches >= self.minimum_matches and (
            label != self.last_announcement
            or now - self.last_announcement_time >= self.repeat_interval
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

                # The general COCO detector sees the portrait printed on banknotes
                # as a person. Currency classification is the more relevant result
                # in this context, so don't report that conflicting detection.
                if label == "person" and state.currency_smoother.note_visible:
                    continue

                if label in LABELS_OF_INTEREST and conf > 0.5:
                    detected.append((label, conf))

        if ULTRALYTICS_AVAILABLE and not state.currency_smoother.note_visible:
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
    latest_frame = queue.Queue(maxsize=1)
    stop_reader = threading.Event()
    stream_error = threading.Event()

    def read_stream():
        buffer = bytearray()
        try:
            with requests.get(stream_url, stream=True, timeout=(5, 5)) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=4096):
                    if stop_reader.is_set():
                        return
                    if not chunk:
                        continue

                    buffer.extend(chunk)
                    while True:
                        start = buffer.find(b"\xff\xd8")
                        end = buffer.find(b"\xff\xd9")
                        if start == -1 or end == -1 or end <= start:
                            if len(buffer) > 1024 * 1024:
                                del buffer[:-2]
                            break

                        jpg = bytes(buffer[start:end + 2])
                        del buffer[:end + 2]
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is None:
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
        except requests.RequestException as error:
            print(f"MJPEG stream request failed: {error}")
            stream_error.set()
        except Exception as error:
            print(f"MJPEG stream reader failed: {error}")
            stream_error.set()

    try:
        probe = requests.get(f"{base_url}/jpg", timeout=5)
        probe.raise_for_status()
    except requests.RequestException as error:
        print(f"Camera probe failed: {error}")
        return False

    # Continuously drain the stream, keeping just one frame so inference always
    # works on the newest camera position instead of processing a stale backlog.
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
                if stream_error.is_set() or failed_reads >= 5:
                    print("MJPEG stream stopped responding.")
                    return False
                continue

            failed_reads = 0
            if not process_frame(frame, object_model, currency_model, state):
                return True
    finally:
        stop_reader.set()
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
