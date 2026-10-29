import os
from settings import config
os.environ["TORCH_CPP_LOG_LEVEL"] = config.TORCH_CPP_LOG_LEVEL
from lib.ReolinkController import ReolinkController
from ultralytics import YOLO
import cv2
import requests
import time
import sys
import numpy as np

import threading
from pathlib import Path
from datetime import datetime, timedelta
import shutil
from collections import deque
import math
from lib.static_filter import StaticObjectFilter
from lib.static_object_mask import StaticObjectMasker
from lib.NotificationWorker import NotificationWorker
import asyncio

# Wszystkie wartości konfiguracyjne: .env.example (PD_*), odczyt w settings.py.
action_policy = config.ACTION_POLICY
RTSP_URL = config.RTSP_URL
BOT_TOKEN = config.TELEGRAM_TOKEN
CHAT_ID = config.TELEGRAM_CHAT_ID


def debug_print(*args, **kwargs):
    if config.DEBUG_LOGS:
        print(*args, **kwargs)


static_filter = StaticObjectFilter(**config.STATIC_OPTIONS)
static_object_mask = StaticObjectMasker(**config.MASK_OPTIONS)
cameraController = ReolinkController(
    host=config.REOLINK_HOST, username=config.REOLINK_USER, password=config.REOLINK_PASSWORD,
    channel=config.REOLINK_CHANNEL,
    retry_seconds=config.REOLINK_RETRY_SECONDS,
    watchdog_seconds=config.REOLINK_WATCHDOG_SECONDS,
    light_seconds=config.LIGHT_DURATION_SECONDS,
    siren_seconds=config.SIREN_DURATION_SECONDS,
    poll_seconds=config.ACTION_POLL_SECONDS,
    light_allowed=lambda: action_policy.allowed("light"),
    siren_allowed=lambda: action_policy.allowed("siren"),
    autostart=config.LIGHT_ENABLED or config.SIREN_ENABLED,
)

MODEL_PATH = config.MODEL_PATH




CONFIDENCE = config.MODEL_CONFIDENCE

model = YOLO(MODEL_PATH)

last_notification = 0
NOTIFICATION_COOLDOWN = config.TELEGRAM_MESSAGE_COOLDOWN_SECONDS

BUFFER_DIR = config.BUFFER_DIR
BUFFER_ANNOTATED_DIR = config.BUFFER_ANNOTATED_DIR
CLEAR_FRAME_PATH = config.CLEAR_DIR
ANNOTATED_FRAME_PATH = config.ANNOTATED_DIR
DAMAGED_FRAME_PATH = config.DAMAGED_DIR

for directory in (BUFFER_DIR, BUFFER_ANNOTATED_DIR, CLEAR_FRAME_PATH, ANNOTATED_FRAME_PATH, DAMAGED_FRAME_PATH):
    directory.mkdir(parents=True, exist_ok=True)

detections = []
lastPxChange = 0
lastEntropy = 0
lastPxChanges = deque(maxlen=config.PIXEL_HISTORY_SIZE)
lastEntropies = deque(maxlen=config.ENTROPY_HISTORY_SIZE)

last_centers = deque(maxlen=config.MOTION_HISTORY_SIZE)
distances = deque(maxlen=config.MOTION_HISTORY_SIZE - 1)

MIN_DETECTIONS = config.ALARM_MIN_DETECTIONS
WINDOW_SECONDS = config.ALARM_WINDOW_SECONDS



# cap = cv2.VideoCapture(RTSP_URL)

class Camera:
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, config.RTSP_BUFFER_SIZE)
        
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.failed_reads = 0
        self.connected_at = time.monotonic()

        t = threading.Thread(target=self.update, daemon=True)
        t.start()

    def update(self):
        while self.running:
            if not action_policy.should_detect():
                with self.lock:
                    self.frame = None
                time.sleep(config.IDLE_SLEEP_SECONDS)
                continue
            ret, frame = self.cap.read()
            
            if time.monotonic() - self.connected_at > config.RTSP_RECONNECT_SECONDS:
                debug_print("RTSP reconnect... every x hours")
                self.reconnect()

                self.connected_at = time.monotonic()
                continue

            if not ret:
                self.failed_reads += 1

                if self.failed_reads > config.RTSP_MAX_FAILED_READS:
                    self.reconnect()
                    self.failed_reads = 0

                continue
  
            self.failed_reads = 0
            with self.lock:
                self.frame = frame
                
    def reconnect(self):
        debug_print("RTSP reconnect...")

        try:
            self.cap.release()
        except:
            pass
            
        while self.running:

            time.sleep(config.RTSP_RETRY_SECONDS)

            # cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap = cv2.VideoCapture(self.url)

            cap.set(cv2.CAP_PROP_BUFFERSIZE, config.RTSP_BUFFER_SIZE)
            ret, frame = cap.read()
            
            if ret :
                debug_print("Reconnect OK")
                
                self.cap = cap
                with self.lock:
                    self.frame = None
                    
                return
            
            debug_print("Reconnect failed...")
            cap.release()

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

camera = Camera(RTSP_URL)
    
def calcPixelChange(frame):
    global lastPxChange
    global lastEntropy
    global lastPxChanges
    global lastEntropies
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    hist = cv2.calcHist([gray], [0], None, [256], [0,256])
    hist = hist / hist.sum()

    lastEntropy = -np.sum(hist * np.log2(hist + 1e-10))
    lastEntropies.append(lastEntropy)
    
    diff = np.abs(np.diff(gray.astype(np.int16), axis=0))
    
    lastPxChange =  np.mean(diff)
    lastPxChanges.append(lastPxChange)
    
    pxChangeDiff = abs((np.mean(lastPxChanges) - lastPxChange))
    
    entropyChangeDiff = abs((np.mean(lastEntropies) - lastEntropy))
    resEntropy = entropyChangeDiff > config.ENTROPY_CHANGE_THRESHOLD
    res = pxChangeDiff > config.PIXEL_CHANGE_THRESHOLD
    
    #debug_print(f"lastEntropies MEAN - {np.mean(lastEntropies)}")
    #debug_print(f"lastEntropy - {lastEntropy}")
    
    if resEntropy == True:
        debug_print(f"lastEntropies MEAN - {np.mean(lastEntropies)}")
        debug_print(f"lastEntropy - {lastEntropy}")
    
    if res == True:
        
        debug_print(f"pixel changes MEAN - {lastPxChanges}")
        debug_print(f"pixel changes MEAN - {np.mean(lastPxChanges)}")
        debug_print(f"pixel changes lastPxChange - {lastPxChange}")
        debug_print(f"pixel changes lastPxChange 1111 - {pxChangeDiff}")
        debug_print(f"lastEntropies MEAN - {np.mean(lastEntropies)}")
        debug_print(f"lastEntropy - {lastEntropy}")
        return True;

    return False
    

def send_photo(photo_path, confidence):
    for attempt in range(config.TELEGRAM_ATTEMPTS):
        delay = config.TELEGRAM_SEND_DELAY_SECONDS if attempt == 0 else config.TELEGRAM_RETRY_SECONDS
        time.sleep(delay)
        # Kolejka mogła czekać, a w międzyczasie skończyły się godziny Telegrama.
        if not action_policy.allowed("telegram"):
            return
        try:
            with open(photo_path, "rb") as photo:
                response = requests.post(
                    f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
                    data={"chat_id": CHAT_ID,
                          "caption": f"Wykryto osobę na podwórku (time= {action_policy.now()}) , confidence= {confidence:.2f}"},
                    files={"photo": photo},
                    timeout=config.TELEGRAM_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
            return
        except requests.RequestException:
            if attempt + 1 >= config.TELEGRAM_ATTEMPTS:
                raise


notification_worker = NotificationWorker(
    send_photo, queue_timeout=config.QUEUE_TIMEOUT_SECONDS
)


def checkIfCanRun(frame):
    global last
        
    scorePxLargeChange = calcPixelChange(frame)

    # debug_print(f"pixel changes in frame - {scorePxLargeChange}")

    if scorePxLargeChange == True:
        debug_print(f"DAMAGED frame")
        filename = timestamp.strftime(
            f"p_%Y_%m_%d_%H_%M_%S_%f_{scorePxLargeChange:.2f}.jpg"
        )
# save damaged frame
        filename = DAMAGED_FRAME_PATH / filename

        cv2.imwrite(str(filename), frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        time.sleep(config.DAMAGED_FRAME_SLEEP_SECONDS)
        return False
        
    return True
# save to buffer

def prune_detections():
    global detections
    cutoff = action_policy.now() - timedelta(seconds=WINDOW_SECONDS)
    recent = [d for d in detections if d["time"] >= cutoff]
    detections, excluded = static_object_mask.filter_alarm_records(recent)
    if excluded:
        print(f"[STATIC_MASK] Wykluczono z alarmu {len(excluded)} wcześniejszych klatek; "
              f"pozostało {len(detections)}/{MIN_DETECTIONS}.", flush=True)


def buff(frame, confidence, results):
    global detections
    timestamp = action_policy.now()
    debug_print(timestamp.strftime(
        "buff_%Y_%m_%d_%H_%M_%S_%f"
    ))

    filename = timestamp.strftime(
        f"p_%Y_%m_%d_%H_%M_%S_%f_{confidence:.2f}_{lastPxChange:.2f}_{lastEntropy:.2f}.jpg"
    )
# Zapisz oryginalną klatkę bez maski i bez ramek YOLO.
    filename_buff = BUFFER_DIR / filename

    cv2.imwrite(str(filename_buff), frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
    
# Zapisz ramki YOLO na obrazie użytym do predykcji, razem z zastosowaną maską.
    filename_buff_annotation = BUFFER_ANNOTATED_DIR / filename

    annotated_frame = results[0].plot(line_width=config.ANNOTATION_LINE_WIDTH, font_size=config.ANNOTATION_FONT_SIZE)
    cv2.imwrite(str(filename_buff_annotation), annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])

    detections.append({
        "time": timestamp,
        "observed_at": time.monotonic(),
        "boxes": [box for r in results for box in r.boxes.xyxy.tolist()],
        "box_confidences": [value for r in results for value in r.boxes.conf.tolist()],
        "file_buff": filename_buff,
        "file_buff_annot": filename_buff_annotation,
        "filename": filename,
        "confidence": confidence

    })
    
    prune_detections()
    
    if len(detections) >= MIN_DETECTIONS:
        meanPxChanges = np.mean(lastPxChanges)
        meanEntropy = np.mean(lastEntropies)
        print("ALARM!")
        debug_print(f" pixel changes  - mean {meanPxChanges:.2f} / last {lastPxChange:.2f}")
        debug_print(f" entropy changes  - mean {meanEntropy:.2f} /  last {lastEntropy:.2f}")

        for i, detection in enumerate(detections):
        
            destination_clear = CLEAR_FRAME_PATH / detection["filename"]

            shutil.move(
                detection["file_buff"],
                destination_clear
            )
            
            destination_annotated = ANNOTATED_FRAME_PATH / detection["filename"]

            shutil.move(
                detection["file_buff_annot"],
                destination_annotated
            )

            if action_policy.allowed("telegram") and (
                not config.TELEGRAM_SEND_FIRST_LAST_ONLY
                or detection == detections[0] or detection == detections[-1]
            ):
                notification_worker.add(
                    (destination_annotated, detection["confidence"])
                )
        if action_policy.allowed("light") or action_policy.allowed("siren"):
            cameraController.alarm()
        detections.clear()

while True:
    timestamp = action_policy.now()
    
    if not action_policy.should_detect():
        time.sleep(config.IDLE_SLEEP_SECONDS)
        continue

    frame = camera.get_frame()
    if frame is None:
        time.sleep(config.NO_FRAME_SLEEP_SECONDS)
        continue
        
    if checkIfCanRun(frame) == False:
        continue

    timestamp = action_policy.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

    filename = f"person_{timestamp}.jpg"
    # debug_print(f"start predict {timestamp} ")
    model_frame = static_object_mask.apply(frame)
    results = model.predict(
        model_frame,
        imgsz=config.MODEL_IMGSZ,
        conf=CONFIDENCE,
        device=config.MODEL_DEVICE,
        classes=config.MODEL_CLASSES,
        iou=config.MODEL_IOU,
        verbose=config.MODEL_VERBOSE
    )

    for i, result in enumerate(results):
        accepted = static_object_mask.observe(result.boxes.xyxy.tolist())
        # Adnotacje i Telegram mają pokazywać dokładnie wejście YOLO, z jego maską.
        # Oryginał bez maski jest osobno przekazywany do buff(frame, ...).
        result.orig_img = model_frame
        results[i] = result[accepted]

    # Cofnij wcześniejsze zliczenia obiektu właśnie uznanego za statyczny,
    # także wtedy, gdy po filtrowaniu bieżąca klatka nie zawiera już wykryć.
    prune_detections()

    detected = False
    confidence = 0

    for r in results:
        if len(r.boxes) > 0:
            detected = True
            # debug_print(f"len e.boxes {len(r.boxes)} ")
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # debug_print(f"len box.xyxy {len(box.xyxy)} ")
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                center = (center_x, center_y)
                confidence = float(box.conf[0])
                
                if static_filter.should_ignore(center):
                    debug_print(f"static filter WORKS !!! centers - {last_centers}")
                    detected = False
                    break
                    
                distances = deque(maxlen=config.MOTION_HISTORY_SIZE - 1)
                last_centers.append(center)
                

                if len(last_centers) > 1:

                    for i in range(1, len(last_centers)):

                        x1, y1 = last_centers[i - 1]
                        x2, y2 = last_centers[i]

                        d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        distances.append(d)
                        debug_print(f"added distance - {d}")
		    
                    avg_distance = sum(distances) / len(distances)
                    debug_print(f"DETECTED- {confidence} / {timestamp} / x={center_x} y={center_y}")
                    debug_print(f"last centers - {last_centers}")
                    debug_print(f" distances - {distances}")
                    debug_print(f" avg_distance - {avg_distance}")
                    lastDist = distances[-1]
                    if lastDist > config.MOTION_MAX_JUMP_PX:
                        debug_print(f"Podejrzany ostatni ruch ")
                        detected = False
                        continue
                    if avg_distance > config.MOTION_MAX_AVERAGE_PX:
                        debug_print(f"Podejrzany ruch duzy skok ")
                        detected = False
                        continue
                    if avg_distance < config.MOTION_MIN_AVERAGE_PX:
                        debug_print(f"Podejrzany ruch maly ruch ")
                        detected = False
                        continue
                
            break

   # cv2.imwrite(CLEAR_FRAME_PATH, frame)

    if detected:
        text = f"{action_policy.now()} Wykryto osobę: conf {confidence:.2f}"
        debug_print(text)
        # send_notification(text)
        buff(frame, confidence, results)
        

    time.sleep(config.LOOP_DELAY_SECONDS)
    
    
    
def send_notification(text):
    global last_notification

    now = time.time()

    if not action_policy.allowed("telegram") or now - last_notification < NOTIFICATION_COOLDOWN:
        return

    requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        data={
            "chat_id": CHAT_ID,
            "text": text
        },
        timeout=config.TELEGRAM_TIMEOUT_SECONDS,
    ).raise_for_status()

    last_notification = now

