import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
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

from settings import RTSP_URL
from settings import REOLINK_IP
from settings import REOLINK_USER
from settings import REOLINK_PASSWORD
from settings import BOT_TOKEN
from settings import CHAT_ID
from settings import HOUR_END


static_filter = StaticObjectFilter()
# Parametry maskowania zmieniaj tutaj; wartości domyślne są też w klasie.
static_object_mask = StaticObjectMasker(
    required_hits=4,          # Ile zgodnych wykryć potrzeba do utworzenia maski (min. 3).
    window_seconds=20,        # Okno historii w sekundach; starsze wykrycia nie są liczone.
    mask_seconds=120,         # Czas życia maski w sekundach, bez przedłużania wykryciami.
    patch_size=15,            # Bok czarnej łaty w pikselach oryginalnej klatki.
    center_tolerance=12,      # Tolerancja położenia i rozrzutu środka ramki w pikselach.
    size_tolerance=0.35,      # Tolerancja zmiany szerokości/wysokości: 0.35 = 35% (min. 4 px).
    movement_threshold=4,    # Spójne przesunięcie od tylu pikseli wyklucza nieruchomość.
    probe_interval=15,       # Co ile sekund YOLO sprawdza klatkę bez maski.
    missing_probes=2,        # Ile kontroli bez obiektu usuwa maskę przed upływem jej czasu.
    mask_iou_threshold=0.65, # IoU już zapamiętanej ramki; niższe = większa tolerancja zmian.
    log_events=True,         # Print przy dodaniu nowej maski (bez wypisywania co klatkę).
)
cameraController = ReolinkController(
    host=REOLINK_IP,
    username=REOLINK_USER,
    password=REOLINK_PASSWORD
)

MODEL_PATH = "best.pt"


DETECTION_START_HOUR = 1
DETECTION_END_HOUR = HOUR_END

CONFIDENCE = 0.50

model = YOLO(MODEL_PATH)

last_notification = 0
NOTIFICATION_COOLDOWN = 2

BUFFER_DIR = Path("./files/buffer")
BUFFER_ANNOTATED_DIR = Path("./files/buff_annot")
CLEAR_FRAME_PATH = Path("./files/clear")
ANNOTATED_FRAME_PATH = Path("./files/annotated")
DAMAGED_FRAME_PATH = Path("./files/damaged")

#BUFFER_DIR.mkdir(exist_ok=True)

detections = []
lastPxChange = 0
lastEntropy = 0
lastPxChanges = deque(maxlen=10)
lastEntropies = deque(maxlen=10)

last_centers = deque(maxlen=5)
distances = deque(maxlen=4)

MIN_DETECTIONS = 6
WINDOW_SECONDS = 60

THRESHOLD_DAMAGED_FRAME = 15

# cap = cv2.VideoCapture(RTSP_URL)

class Camera:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.failed_reads = 0
        self.connected_at = time.time()

        t = threading.Thread(target=self.update, daemon=True)
        t.start()

    def update(self):
        while self.running:
            current_hour = datetime.now().hour
            if not (DETECTION_START_HOUR <= current_hour < DETECTION_END_HOUR):
                time.sleep(60)
                continue
            ret, frame = self.cap.read()
            
            if time.time() - self.connected_at > 6 * 3600:
                print("RTSP reconnect... every x hours")
                self.reconnect()

                self.connected_at = time.time()
                continue

            if not ret:
                self.failed_reads += 1

                if self.failed_reads > 5:
                    self.reconnect()
                    self.failed_reads = 0

                continue
  
            self.failed_reads = 0
            with self.lock:
                self.frame = frame
                
    def reconnect(self):
        print("RTSP reconnect...")

        try:
            self.cap.release()
        except:
            pass
            
        while self.running:

            time.sleep(2)

            # cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap = cv2.VideoCapture(RTSP_URL)

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, frame = cap.read()
            
            if ret :
                print("Reconnect OK")
                
                self.cap = cap
                with self.lock:
                    self.frame = None
                    
                return
            
            print("Reconnect failed...")
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
    resEntropy = entropyChangeDiff > 1
    res = pxChangeDiff > 1.6
    
    #print(f"lastEntropies MEAN - {np.mean(lastEntropies)}")
    #print(f"lastEntropy - {lastEntropy}")
    
    if resEntropy == True:
        print(f"lastEntropies MEAN - {np.mean(lastEntropies)}")
        print(f"lastEntropy - {lastEntropy}")
    
    if res == True:
        
        print(f"pixel changes MEAN - {lastPxChanges}")
        print(f"pixel changes MEAN - {np.mean(lastPxChanges)}")
        print(f"pixel changes lastPxChange - {lastPxChange}")
        print(f"pixel changes lastPxChange 1111 - {pxChangeDiff}")
        print(f"lastEntropies MEAN - {np.mean(lastEntropies)}")
        print(f"lastEntropy - {lastEntropy}")
        return True;

    return False
    

def send_photo(photo_path, confidence):
    with open(photo_path, "rb") as photo:
        try:
            time.sleep(0.1)
            requests.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
                data={
                    "chat_id": CHAT_ID,
                    "caption": f"Wykryto osobę na podwórku (time= {datetime.now()}) , confidence= {confidence:.2f}"
                },
                files={
                    "photo": photo
                }
            )
        except:
            time.sleep(1)
            requests.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
                data={
                    "chat_id": CHAT_ID,
                    "caption": f"Wykryto osobę na podwórku (time= {datetime.now()}) , confidence= {confidence:.2f}"
                },
                files={
                    "photo": photo
                }
            )

notification_worker = NotificationWorker(
    send_photo
)


def checkIfCanRun(frame):
    global last
        
    scorePxLargeChange = calcPixelChange(frame)

    # print(f"pixel changes in frame - {scorePxLargeChange}")

    if scorePxLargeChange == True:
        print(f"DAMAGED frame")
        filename = timestamp.strftime(
            f"p_%Y_%m_%d_%H_%M_%S_%f_{scorePxLargeChange:.2f}.jpg"
        )
# save damaged frame
        filename = DAMAGED_FRAME_PATH / filename

        cv2.imwrite(str(filename), frame)
        time.sleep(1)
        return False
        
    return True
# save to buffer

def prune_detections():
    global detections
    cutoff = datetime.now() - timedelta(seconds=WINDOW_SECONDS)
    recent = [d for d in detections if d["time"] >= cutoff]
    detections, excluded = static_object_mask.filter_alarm_records(recent)
    if excluded:
        print(f"[STATIC_MASK] Wykluczono z alarmu {len(excluded)} wcześniejszych klatek; "
              f"pozostało {len(detections)}/{MIN_DETECTIONS}.", flush=True)


def buff(frame, confidence, results):
    global detections
    timestamp = datetime.now()
    print(timestamp.strftime(
        "buff_%Y_%m_%d_%H_%M_%S_%f"
    ))

    filename = timestamp.strftime(
        f"p_%Y_%m_%d_%H_%M_%S_%f_{confidence:.2f}_{lastPxChange:.2f}_{lastEntropy:.2f}.jpg"
    )
# Zapisz oryginalną klatkę bez maski i bez ramek YOLO.
    filename_buff = BUFFER_DIR / filename

    cv2.imwrite(str(filename_buff), frame)
    
# Zapisz ramki YOLO na obrazie użytym do predykcji, razem z zastosowaną maską.
    filename_buff_annotation = BUFFER_ANNOTATED_DIR / filename

    annotated_frame = results[0].plot(line_width=1,font_size=1)
    cv2.imwrite(str(filename_buff_annotation), annotated_frame)

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
        print(f" pixel changes  - mean {meanPxChanges:.2f} / last {lastPxChange:.2f}")
        print(f" entropy changes  - mean {meanEntropy:.2f} /  last {lastEntropy:.2f}")

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

            if detection == detections[0] or detection == detections[-1]:
                notification_worker.add(
                    (destination_annotated, detection["confidence"])
                )
        cameraController.alarm()
        detections.clear()

while True:
    timestamp = datetime.now()
    
    current_hour = datetime.now().hour

    if not (DETECTION_START_HOUR <= current_hour < DETECTION_END_HOUR):
        time.sleep(60)
        continue

    frame = camera.get_frame()
    if frame is None:
        time.sleep(0.8)
        continue
        
    if checkIfCanRun(frame) == False:
        continue

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

    filename = f"person_{timestamp}.jpg"
    # print(f"start predict {timestamp} ")
    model_frame = static_object_mask.apply(frame)
    results = model.predict(
        model_frame,
        imgsz=960,
        conf=CONFIDENCE,
        device="cpu",
        classes=[0],
        iou=0.5,
        verbose=False
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
            # print(f"len e.boxes {len(r.boxes)} ")
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # print(f"len box.xyxy {len(box.xyxy)} ")
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                center = (center_x, center_y)
                confidence = float(box.conf[0])
                
                if static_filter.should_ignore(center):
                    print(f"static filter WORKS !!! centers - {last_centers}")
                    detected = False
                    break
                    
                distances = deque(maxlen=5)
                last_centers.append(center)
                

                if len(last_centers) > 1:

                    for i in range(1, len(last_centers)):

                        x1, y1 = last_centers[i - 1]
                        x2, y2 = last_centers[i]

                        d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        distances.append(d)
                        print(f"added distance - {d}")
		    
                    avg_distance = sum(distances) / len(distances)
                    print(f"DETECTED- {confidence} / {timestamp} / x={center_x} y={center_y}")
                    print(f"last centers - {last_centers}")
                    print(f" distances - {distances}")
                    print(f" avg_distance - {avg_distance}")
                    lastDist = distances[-1]
                    if lastDist > 550:
                        print(f"Podejrzany ostatni ruch ")
                        detected = False
                        continue
                    if avg_distance > 480 :
                        print(f"Podejrzany ruch duzy skok ")
                        detected = False
                        continue
                    if avg_distance < 3:
                        print(f"Podejrzany ruch maly ruch ")
                        detected = False
                        continue
                
            break

   # cv2.imwrite(CLEAR_FRAME_PATH, frame)

    if detected:
        text = f"{datetime.now()} Wykryto osobę: conf {confidence:.2f}"
        print(text)
        # send_notification(text)
        buff(frame, confidence, results)
        

    time.sleep(0.3)
    
    
    
def send_notification(text):
    global last_notification

    now = time.time()

    if now - last_notification < NOTIFICATION_COOLDOWN:
        return

    requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        data={
            "chat_id": CHAT_ID,
            "text": text
        }
    )

    last_notification = now

