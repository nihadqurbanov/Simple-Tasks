"""
Azərbaycan Qaraj Girişi — GUI Versiyası (Yaddaşlı)
═══════════════════════════════════════════════════════════
DƏYIŞIKLIKLƏR:
  ✅ Maşın məlumatları həmişə yadda qalır (silinmir)
  ✅ Zonalar: 75%, 70%, 65%, 60%, 55%, 50%, 45%, 40%, 35%, 30%
  ✅ ID nömrələri artmağa davam edir (reset olmur)

MANTIQ:
  🔵 Mavi xətt — yalnız YUXARIDAN→AŞAĞI keçiş tanınır (aşağıdan 75%)
       → top-edge (y1) əvvəl üstdə olub, sonra mavi Y-yə çatdı
       → Sonra 10 zonada frame topla (2×10 = 20)
  🟢 Yaşıl xətt — x2 >= green_x → qaraj girişi
  🎯 OCR — toplanan frame-lar ilə cəhd
  ⚠️  w/h ≤ 1.5 olan maşınlar (şaquli) filtrlənir
  📍 Zonalar: 75%, 70%, 65%, 60%, 55%, 50%, 45%, 40%, 35%, 30%
"""

import cv2
import numpy as np
import re
import os
import sys
import logging
import threading
import queue
from datetime import datetime
from tkinter import Tk, Frame, Label, Canvas, Text, Scrollbar, END, PhotoImage, StringVar
from PIL import Image, ImageTk

os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['OPENCV_FFMPEG_LOG_LEVEL'] = '-8'
logging.getLogger('easyocr').setLevel(logging.ERROR)

import warnings

warnings.filterwarnings('ignore')

from ultralytics import YOLO
import easyocr


# ═══════════════════════════════════════════════════════════════════
#  BACKEND
# ═══════════════════════════════════════════════════════════════════

class GarageBackend:
    def __init__(self, rtsp_url, frame_queue: queue.Queue, log_queue: queue.Queue):
        self.rtsp_url = rtsp_url
        self.frame_q = frame_queue
        self.log_q = log_queue
        self.running = True

        # ── Modellər ──
        self.log("SİSTEM", "YOLOv8 yüklənir...")
        self.model = YOLO('yolov8n.pt')
        self.log("SİSTEM", "✅ YOLOv8 hazırdır")

        self.log("SİSTEM", "EasyOCR yüklənir...")
        self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        self.log("SİSTEM", "✅ EasyOCR hazırdır")

        # ── Virtual xəttlər ──
        self.blue_y = None
        self.zone_75 = None
        self.zone_70 = None
        self.zone_65 = None
        self.zone_60 = None
        self.zone_55 = None
        self.zone_50 = None
        self.zone_45 = None
        self.zone_40 = None
        self.zone_35 = None
        self.zone_30 = None
        self.green_x = None

        # ── Tracking ──
        self.tracked = {}
        self.next_id = 0

        # ── Arxiv (silinmiş maşınlar) ──
        self.archived = {}

        # ── Qovluqlar ──
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.dir_cars = os.path.join(self.base_dir, 'cars_frames')
        os.makedirs(self.dir_cars, exist_ok=True)

        # ── CSV ──
        self.csv_path = os.path.join(self.base_dir, 'qaraj_log.csv')
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', encoding='utf-8') as f:
                f.write("Tarix,ID,Nomre,Guven\n")

    # ─── Log ──────────────────────────────────────────────────────
    def log(self, tag, msg):
        self.log_q.put((tag, msg))

    # ─── cars_frames/id_X/ qovluq helper ─────────────────────────
    def _get_car_dirs(self, vid):
        """
        cars_frames/
          id_0/
            originals_frames/   ← plate crop
            fullcar_frames/     ← tam maşın
        Qovluqları yarat, path-ları qaytar
        """
        base = os.path.join(self.dir_cars, f"id_{vid}")
        originals = os.path.join(base, 'originals_frames')
        fullcars = os.path.join(base, 'fullcar_frames')
        os.makedirs(originals, exist_ok=True)
        os.makedirs(fullcars, exist_ok=True)
        return originals, fullcars

    # ─── Setup xəttlər ───────────────────────────────────────────
    def setup_lines(self, h, w):
        self.blue_y = int(h * 0.25)  # mavi xətt (top 25% = aşağıdan 75%)
        self.zone_75 = int(h * 0.25)
        self.zone_70 = int(h * 0.30)
        self.zone_65 = int(h * 0.35)
        self.zone_60 = int(h * 0.40)
        self.zone_55 = int(h * 0.45)
        self.zone_50 = int(h * 0.50)
        self.zone_45 = int(h * 0.55)
        self.zone_40 = int(h * 0.60)
        self.zone_35 = int(h * 0.65)
        self.zone_30 = int(h * 0.70)
        self.green_x = int(w * 0.80)
        self.log("SİSTEM", f"📏 Garajın önü Y=25% | Qaraja giriş X=80% | 10 zona hazır")

    # ─── Plate validation ─────────────────────────────────────────
    def is_valid_plate(self, text):
        if not text or len(text) < 7:
            return None
        t = re.sub(r'[^A-Z0-9\s\-]', '', text.upper().strip())
        if re.match(r'^[0-9]{2}[\s\-]?[A-Z]{2}[\s\-]?[0-9]{3}$', t):
            return t
        return None

    def format_plate(self, text):
        if not text:
            return None
        text = re.sub(r'[^A-Z0-9]', '', text.upper().strip())

        repl = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', '|': '1', 'J': '1',
                'S': '5', 'Z': '2', 'G': '6', 'B': '8', 'T': '7', 'E': '3'}

        if len(text) < 7:
            return None

        first = ''
        for c in text[:3]:
            if c.isdigit():
                first += c
            elif c in repl and repl[c].isdigit():
                first += repl[c]
            if len(first) == 2:
                break

        last = ''
        for c in reversed(text[-4:]):
            if c.isdigit():
                last = c + last
            elif c in repl and repl[c].isdigit():
                last = repl[c] + last
            if len(last) == 3:
                break

        if len(first) != 2 or len(last) != 3:
            return None

        mid_raw = text[len(first):-len(last)]
        mid = ''
        for c in mid_raw:
            if c.isalpha():
                mid += c
            elif c == '0':
                mid += 'O'
            elif c == '1':
                mid += 'I'

        if len(mid) == 2:
            plate = f"{first}-{mid}-{last}"
            if self.is_valid_plate(plate):
                return plate
        return None

    # ─── Zona tapma ───────────────────────────────────────────────
    def get_plate_zone(self, bbox):
        """y2 (aşağı edge) əsasında zona qur"""
        _, _, _, y2 = bbox
        if y2 >= self.zone_30:
            return '30%'
        elif y2 >= self.zone_35:
            return '35%'
        elif y2 >= self.zone_40:
            return '40%'
        elif y2 >= self.zone_45:
            return '45%'
        elif y2 >= self.zone_50:
            return '50%'
        elif y2 >= self.zone_55:
            return '55%'
        elif y2 >= self.zone_60:
            return '60%'
        elif y2 >= self.zone_65:
            return '65%'
        elif y2 >= self.zone_70:
            return '70%'
        elif y2 >= self.zone_75:
            return '75%'
        return None

    # ─── OCR ──────────────────────────────────────────────────────
    def preprocess(self, img):
        h, w = img.shape[:2]
        if h < 300:
            s = 300 / h
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enh = clahe.apply(gray)
        den = cv2.bilateralFilter(enh, 15, 25, 25)
        return cv2.cvtColor(den, cv2.COLOR_GRAY2BGR)

    def ocr_single(self, plate_img, full_car):
        h, w = plate_img.shape[:2]
        if h < 150:
            s = 200 / h
            plate_img = cv2.resize(plate_img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)

        candidates = []
        try:
            for (_, text, conf) in self.reader.readtext(
                    plate_img, detail=1,
                    allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', paragraph=False):
                if conf > 0.25:
                    candidates.append((text, conf))

            for (_, text, conf) in self.reader.readtext(
                    self.preprocess(plate_img), detail=1,
                    allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', paragraph=False):
                if conf > 0.25:
                    candidates.append((text, conf))

            for (_, text, conf) in self.reader.readtext(
                    full_car, detail=1,
                    allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', paragraph=False):
                if conf > 0.20:
                    candidates.append((text, conf))

            candidates.sort(key=lambda x: (len(x[0]), x[1]), reverse=True)
            for text, conf in candidates:
                plate = self.format_plate(text)
                if plate:
                    return plate, conf
        except Exception:
            pass
        return None, 0

    # ─── Detect vehicles ──────────────────────────────────────────
    def detect_vehicles(self, frame):
        results = self.model(frame, conf=0.3, classes=[2, 5, 7], verbose=False)
        out = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                car = frame[y1:y2, x1:x2]
                if car.size == 0:
                    continue
                h, w = car.shape[:2]

                # ⚠️ w/h ≤ 1.3 filtri
                aspect_ratio = (w / h) if h > 0 else 0
                if aspect_ratio >= 1.3:
                    continue

                plate_crop = car[int(h * 0.50):h, :, :]
                if plate_crop.size == 0:
                    continue
                out.append({
                    'bbox': (x1, y1, x2, y2),
                    'plate_img': plate_crop,
                    'full_car': car,
                    'conf': conf
                })
        return out

    # ─── ID match ─────────────────────────────────────────────────
    def match_id(self, bbox):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        best_id, best_d = None, float('inf')
        for vid, vd in self.tracked.items():
            px, py = vd['last_pos']
            d = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
            if d < 200 and d < best_d:
                best_d = d
                best_id = vid
        return best_id

    # ─── Ana loop ─────────────────────────────────────────────────
    def run(self):
        if self.rtsp_url.startswith('rtsp://'):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = \
                "rtsp_transport;tcp|rtsp_flags;prefer_tcp|loglevel;fatal"
            import contextlib
            with contextlib.redirect_stderr(None):
                cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(self.rtsp_url)

        if not cap.isOpened():
            self.log("XƏTA", f"❌ Kameraya qoşulmaq olmadı: {self.rtsp_url}")
            return

        ret, first = cap.read()
        if ret:
            h, w = first.shape[:2]
            self.setup_lines(h, w)
        else:
            self.log("XƏTA", "❌ İlk frame alına bilmədi")
            return

        self.log("SİSTEM", "🎥 Kamera aktiv")

        frame_count = 0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1
            if frame_count % 3 != 0:
                continue

            vehicles = self.detect_vehicles(frame)

            for v in vehicles:
                bbox = v['bbox']
                x1, y1, x2, y2 = bbox
                plate_img = v['plate_img']
                full_car = v['full_car']
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                vid = self.match_id(bbox)

                # ── Yeni maşın: ID yalnız ÜSTDƏN GƏLƏNLƏRƏ verilir ──
                # y1 (top edge) mavi xəttin üstündə olmalı → aşağıya hərəkət etməkdə
                if vid is None:
                    if y1 < self.blue_y:
                        # Maşın hələ mavi xəttın üstündə → yeni ID ver
                        vid = self.next_id
                        self.next_id += 1
                        self.tracked[vid] = {
                            'last_pos': (cx, cy),
                            'prev_y1': y1,
                            'entered_blue': False,
                            'in_garage': False,
                            'ocr_done': False,
                            'frames_75': [],
                            'frames_70': [],
                            'frames_65': [],
                            'frames_60': [],
                            'frames_55': [],
                            'frames_50': [],
                            'frames_45': [],
                            'frames_40': [],
                            'frames_35': [],
                            'frames_30': [],
                            'plate': None,
                            'last_seen': frame_count,
                            'logged_zones': set(),
                            'blue_frame': None,
                        }
                    else:
                        continue

                if vid not in self.tracked:
                    continue

                vd = self.tracked[vid]
                vd['last_pos'] = (cx, cy)
                vd['last_seen'] = frame_count

                # ═══ MAVI XƏTT: yuxarı→aşağı keçiş aşkarla ═══
                if not vd['entered_blue']:
                    if vd['prev_y1'] < self.blue_y and y1 >= self.blue_y:
                        vd['entered_blue'] = True
                        vd['blue_frame'] = (plate_img.copy(), full_car.copy())
                        orig_dir, full_dir = self._get_car_dirs(vid)
                        ts = datetime.now().strftime("%H%M%S_%f")
                        cv2.imwrite(os.path.join(orig_dir, f"blue_{ts}.jpg"),
                                    plate_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        cv2.imwrite(os.path.join(full_dir, f"blue_{ts}.jpg"),
                                    full_car, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        self.log("MAVİ", f"[ID:{vid}] 🔵 Mavi zonaya daxil oldu (↓ keçiş)")
                    vd['prev_y1'] = y1

                # ═══ ZONA FRAME TOPLAMA (yalnız mavi keçiş sonra) ═══
                if vd['entered_blue'] and not vd['in_garage']:
                    zone = self.get_plate_zone(bbox)

                    if zone and zone not in vd['logged_zones']:
                        vd['logged_zones'].add(zone)
                        self.log("TOPLA", f"[ID:{vid}] 📍 {zone} zonasına daxil")

                    zone_key = {
                        '75%': 'frames_75', '70%': 'frames_70',
                        '65%': 'frames_65', '60%': 'frames_60',
                        '55%': 'frames_55', '50%': 'frames_50',
                        '45%': 'frames_45', '40%': 'frames_40',
                        '35%': 'frames_35', '30%': 'frames_30',
                    }
                    if zone and zone in zone_key:
                        key = zone_key[zone]
                        if len(vd[key]) < 2:
                            vd[key].append((plate_img.copy(), full_car.copy()))
                            orig_dir, full_dir = self._get_car_dirs(vid)
                            ts = datetime.now().strftime("%H%M%S_%f")
                            zone_tag = zone.replace('%', '')
                            cv2.imwrite(os.path.join(orig_dir, f"zone_{zone_tag}_{ts}.jpg"),
                                        plate_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            cv2.imwrite(os.path.join(full_dir, f"zone_{zone_tag}_{ts}.jpg"),
                                        full_car, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # ═══ YAŞIL XƏTT: x2 >= green_x → qaraj girişi ═══
                if vd['entered_blue'] and not vd['in_garage'] and x2 >= self.green_x:
                    vd['in_garage'] = True
                    total = sum(len(vd[k]) for k in
                                ('frames_75', 'frames_70', 'frames_65', 'frames_60', 'frames_55',
                                 'frames_50', 'frames_45', 'frames_40', 'frames_35', 'frames_30'))
                    self.log("QARAJ", f"[ID:{vid}] 🟢 QARAJA DAXİL OLDU ({total} frame)")

                    if total == 0:
                        orig_dir, full_dir = self._get_car_dirs(vid)

                        if vd['blue_frame']:
                            vd['frames_75'].append(vd['blue_frame'])
                            ts = datetime.now().strftime("%H%M%S_%f")
                            cv2.imwrite(os.path.join(orig_dir, f"blue_fallback_{ts}.jpg"),
                                        vd['blue_frame'][0], [cv2.IMWRITE_JPEG_QUALITY, 95])
                            cv2.imwrite(os.path.join(full_dir, f"blue_fallback_{ts}.jpg"),
                                        vd['blue_frame'][1], [cv2.IMWRITE_JPEG_QUALITY, 95])

                        vd['frames_30'].append((plate_img.copy(), full_car.copy()))
                        ts = datetime.now().strftime("%H%M%S_%f")
                        cv2.imwrite(os.path.join(orig_dir, f"green_{ts}.jpg"),
                                    plate_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        cv2.imwrite(os.path.join(full_dir, f"green_{ts}.jpg"),
                                    full_car, [cv2.IMWRITE_JPEG_QUALITY, 95])

                        self.log("QARAJ", f"[ID:{vid}] ⚠️ Zona frame yox idi, mavi + yaşıl frame əlavə edildi")

                # ═══ OCR (tək dəfə) ═══
                if vd['in_garage'] and not vd['ocr_done']:
                    vd['ocr_done'] = True
                    threading.Thread(target=self._run_ocr, args=(vid,), daemon=True).start()

            # ── Annotasiya ──
            annotated = self._draw(frame, vehicles)

            # ── Köhnə maşınları arxivə köçür (amma SİLMƏ) ──
            if frame_count % 150 == 0:
                to_archive = [vid for vid, vd in self.tracked.items()
                              if frame_count - vd['last_seen'] > 6000]
                for vid in to_archive:
                    self.archived[vid] = self.tracked[vid]
                    del self.tracked[vid]
                    self.log("SİSTEM", f"[ID:{vid}] 📦 Arxivə köçürüldü")

            # ── GUI-yə frame ──
            try:
                self.frame_q.put_nowait(annotated)
            except queue.Full:
                pass

        cap.release()
        self.log("SİSTEM", "✅ Kamera bağlandı")

    # ─── OCR thread ───────────────────────────────────────────────
    def _run_ocr(self, vid):
        vd = self.tracked.get(vid)
        if not vd:
            return

        orig_dir, full_dir = self._get_car_dirs(vid)

        all_frames = (vd['frames_75'] + vd['frames_70'] + vd['frames_65'] + vd['frames_60'] +
                      vd['frames_55'] + vd['frames_50'] + vd['frames_45'] + vd['frames_40'] +
                      vd['frames_35'] + vd['frames_30'])
        self.log("OCR", f"[ID:{vid}] 🔄 OCR başlayır ({len(all_frames)} frame)")

        for i, (p_img, c_img) in enumerate(all_frames, 1):
            plate, conf = self.ocr_single(p_img, c_img)

            ts = datetime.now().strftime("%H%M%S_%f")
            cv2.imwrite(os.path.join(orig_dir, f"ocr_cehd{i}_{ts}.jpg"),
                        p_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(os.path.join(full_dir, f"ocr_cehd{i}_{ts}.jpg"),
                        c_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            if plate:
                vd['plate'] = plate
                self.log("OCR", f"[ID:{vid}] ✅ Nömrə: {plate} (cəhd {i}, güvən {conf:.0%})")

                cv2.imwrite(os.path.join(orig_dir, f"tanindi_{plate}_{ts}.jpg"),
                            p_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(os.path.join(full_dir, f"tanindi_{plate}_{ts}.jpg"),
                            c_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(self.csv_path, 'a', encoding='utf-8') as f:
                    f.write(f"{now},{vid},{plate},{conf:.2f}\n")

                self.log("Nəticə", f"[ID:{vid}] → {plate}")
                return

            else:
                self.log("OCR", f"[ID:{vid}] ❌ Cəhd {i}/{len(all_frames)} – tapılmadı")

        self.log("OCR", f"[ID:{vid}] ❌ {len(all_frames)} cəhd sonra tanınmadı")

        if all_frames:
            last_p, last_c = all_frames[-1]
            ts = datetime.now().strftime("%H%M%S_%f")
            cv2.imwrite(os.path.join(orig_dir, f"taninmadi_{ts}.jpg"),
                        last_p, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(os.path.join(full_dir, f"taninmadi_{ts}.jpg"),
                        last_c, [cv2.IMWRITE_JPEG_QUALITY, 95])

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.csv_path, 'a', encoding='utf-8') as f:
            f.write(f"{now},{vid},TANINDI_DEYIL,0.00\n")

        self.log("Nəticə", f"[ID:{vid}] → Tanınmadı")

    # ─── Draw ─────────────────────────────────────────────────────
    def _draw(self, frame, vehicles):
        out = frame.copy()
        h, w = out.shape[:2]

        # Mavi xətt
        cv2.line(out, (0, self.blue_y), (w, self.blue_y), (255, 100, 0), 3)
        cv2.putText(out, "BLUE 75%", (10, self.blue_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 0), 2)

        # Zona xəttləri (dashed)
        zones = [
            (self.zone_70, "70%"), (self.zone_65, "65%"), (self.zone_60, "60%"),
            (self.zone_55, "55%"), (self.zone_50, "50%"), (self.zone_45, "45%"),
            (self.zone_40, "40%"), (self.zone_35, "35%"), (self.zone_30, "30%")
        ]
        for zy, zl in zones:
            for x in range(0, w, 30):
                cv2.line(out, (x, zy), (min(x + 15, w), zy), (180, 180, 180), 1)
            cv2.putText(out, zl, (10, zy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # Yaşıl xətt
        cv2.line(out, (self.green_x, 0), (self.green_x, h), (0, 255, 0), 3)
        cv2.putText(out, "GREEN", (self.green_x + 6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # Bbox + ID
        for v in vehicles:
            x1, y1, x2, y2 = v['bbox']
            vid = self.match_id(v['bbox'])

            color = (255, 255, 255)
            if vid is not None and vid in self.tracked:
                vd = self.tracked[vid]
                if vd['in_garage']:
                    color = (0, 255, 100)
                elif vd['entered_blue']:
                    color = (255, 150, 0)

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            label = f"ID:{vid}" if vid is not None else "?"
            if vid is not None and vid in self.tracked and self.tracked[vid].get('plate'):
                label += f" {self.tracked[vid]['plate']}"

            cv2.rectangle(out, (x1, y1 - 22), (x1 + len(label) * 11 + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return out


# ═══════════════════════════════════════════════════════════════════
#  GUI
# ═══════════════════════════════════════════════════════════════════

class GarageGUI:
    BG_DARK = "#0f1117"
    BG_PANEL = "#161922"
    BG_LOG = "#1a1d2e"
    ACCENT_BLUE = "#4f8cff"
    ACCENT_GRN = "#2ecc71"
    ACCENT_RED = "#e74c3c"
    TEXT_MAIN = "#e8eaf0"
    TEXT_DIM = "#6b7280"

    TAG_COLORS = {
        "SİSTEM": "#6b7280",
        "MAVİ": "#4f8cff",
        "TOPLA": "#a78bfa",
        "QARAJ": "#2ecc71",
        "OCR": "#f39c12",
        "SONUÇ": "#2ecc71",
        "XƏTA": "#e74c3c",
    }

    def __init__(self, root, rtsp_url):
        self.root = root
        self.root.title("🇦🇿 Azərbaycan Qaraj Sistemi (Yaddaşlı)")
        self.root.configure(bg=self.BG_DARK)
        self.root.geometry("1280x760")
        self.root.minsize(1100, 660)

        self.frame_q = queue.Queue(maxsize=2)
        self.log_q = queue.Queue(maxsize=200)

        self.backend = GarageBackend(rtsp_url, self.frame_q, self.log_q)
        self._build_ui()

        self.backend_thread = threading.Thread(target=self.backend.run, daemon=True)
        self.backend_thread.start()
        self._refresh()

    def _build_ui(self):
        # Top bar
        top = Frame(self.root, bg=self.BG_PANEL, pady=8)
        top.pack(fill='x', side='top')
        Label(top, text="🇦🇿  QARAJ SİSTEMİ (YADDAŞLI)", font=("Courier New", 18, "bold"),
              bg=self.BG_PANEL, fg=self.ACCENT_BLUE).pack(side='left', padx=16)
        self.status_var = StringVar(value="⏳ Yüklənir...")
        Label(top, textvariable=self.status_var, font=("Courier New", 11),
              bg=self.BG_PANEL, fg=self.TEXT_DIM).pack(side='right', padx=16)

        # Main
        main = Frame(self.root, bg=self.BG_DARK)
        main.pack(fill='both', expand=True, padx=8, pady=8)

        # Left: video
        left = Frame(main, bg=self.BG_DARK)
        left.pack(side='left', fill='both', expand=True)

        legend = Frame(left, bg=self.BG_PANEL, pady=4)
        legend.pack(fill='x', pady=(0, 4))
        for color, text in [
            (self.ACCENT_BLUE, "● Mavi zona (75%)"),
            ("#a78bfa", "● Zonalar (70%→30%)"),
            (self.ACCENT_GRN, "● Yaşıl xətt (qaraj girişi)"),
        ]:
            Label(legend, text=text, font=("Courier New", 9),
                  bg=self.BG_PANEL, fg=color).pack(side='left', padx=10)

        self.canvas = Canvas(left, bg="black", highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)

        # Right: log
        right = Frame(main, bg=self.BG_DARK, width=500)
        right.pack(side='right', fill='y', padx=(8, 0))
        right.pack_propagate(False)

        log_header = Frame(right, bg=self.BG_PANEL, pady=6)
        log_header.pack(fill='x')
        Label(log_header, text="📋  SİSTEM LOG", font=("Courier New", 13, "bold"),
              bg=self.BG_PANEL, fg=self.TEXT_MAIN).pack(side='left', padx=12)
        self.log_count_var = StringVar(value="0")
        Label(log_header, textvariable=self.log_count_var, font=("Courier New", 10),
              bg=self.BG_PANEL, fg=self.TEXT_DIM).pack(side='right', padx=12)

        log_frame = Frame(right, bg=self.BG_DARK)
        log_frame.pack(fill='both', expand=True)

        self.log_text = Text(log_frame, bg=self.BG_LOG, fg=self.TEXT_MAIN,
                             font=("Courier New", 9), wrap='word',
                             state='disabled', bd=0, padx=10, pady=6,
                             highlightthickness=0)
        self.log_text.pack(side='left', fill='both', expand=True)

        scrollbar = Scrollbar(log_frame, orient='vertical', command=self.log_text.yview,
                              bg=self.BG_DARK, troughcolor=self.BG_LOG,
                              activebackground=self.ACCENT_BLUE)
        scrollbar.pack(side='right', fill='y')
        self.log_text.config(yscrollcommand=scrollbar.set)

        for tag, color in self.TAG_COLORS.items():
            self.log_text.tag_configure(tag, foreground=color)
        self.log_text.tag_configure("TIME", foreground=self.TEXT_DIM)
        self.log_text.tag_configure("MSG", foreground=self.TEXT_MAIN)
        self.log_text.tag_configure("SONUC_OK", foreground=self.ACCENT_GRN, font=("Courier New", 10, "bold"))
        self.log_text.tag_configure("SONUC_NO", foreground=self.ACCENT_RED, font=("Courier New", 10, "bold"))

        self.log_line_count = 0

        # Bottom stats
        stats = Frame(self.root, bg=self.BG_PANEL, pady=6)
        stats.pack(fill='x', side='bottom')
        self.stat_entries = StringVar(value="0")
        self.stat_plates = StringVar(value="0")
        self.stat_active = StringVar(value="0")
        self.stat_total = StringVar(value="0")
        for label, var in [("🚗 Giriş:", self.stat_entries),
                           ("🏷️  Tanınan:", self.stat_plates),
                           ("📍 Aktiv:", self.stat_active),
                           ("💾 Yaddaş:", self.stat_total)]:
            f = Frame(stats, bg=self.BG_PANEL)
            f.pack(side='left', padx=18)
            Label(f, text=label, font=("Courier New", 10),
                  bg=self.BG_PANEL, fg=self.TEXT_DIM).pack(side='left')
            Label(f, textvariable=var, font=("Courier New", 12, "bold"),
                  bg=self.BG_PANEL, fg=self.TEXT_MAIN).pack(side='left', padx=(4, 0))

    def _refresh(self):
        # Video
        try:
            frame = self.frame_q.get_nowait()
            cw = self.canvas.winfo_width() or 820
            ch = self.canvas.winfo_height() or 580
            if cw < 100: cw, ch = 820, 580

            h, w = frame.shape[:2]
            scale = min(cw / w, ch / h)
            nw, nh = int(w * scale), int(h * scale)
            resized = cv2.resize(frame, (nw, nh))
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(resized)
            self._photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image((cw - nw) // 2, (ch - nh) // 2, anchor='nw', image=self._photo)
            self.status_var.set("🟢 Canlı")
        except queue.Empty:
            pass

        # Log
        while not self.log_q.empty():
            try:
                tag, msg = self.log_q.get_nowait()
            except queue.Empty:
                break

            self.log_line_count += 1
            time_str = datetime.now().strftime("%H:%M:%S")

            self.log_text.config(state='normal')

            is_sonuc_ok = (tag == "SONUÇ" and "→" in msg and "Tanınmadı" not in msg)
            is_sonuc_no = (tag == "SONUÇ" and "Tanınmadı" in msg)

            if tag == "SONUÇ":
                self.log_text.insert(END, "─" * 52 + "\n", "TIME")

            self.log_text.insert(END, f"[{time_str}] ", "TIME")
            tag_color = tag if tag in self.TAG_COLORS else "MSG"
            self.log_text.insert(END, f"[{tag:>6}] ", tag_color)

            if is_sonuc_ok:
                self.log_text.insert(END, msg + "\n", "SONUC_OK")
            elif is_sonuc_no:
                self.log_text.insert(END, msg + "\n", "SONUC_NO")
            else:
                self.log_text.insert(END, msg + "\n", "MSG")

            self.log_text.config(state='disabled')
            self.log_text.see(END)
            self.log_count_var.set(str(self.log_line_count))

        # Stats
        entries = sum(1 for vd in self.backend.tracked.values() if vd['in_garage'])
        plates = sum(1 for vd in self.backend.tracked.values() if vd.get('plate'))
        active = len(self.backend.tracked)
        total = len(self.backend.tracked) + len(self.backend.archived)

        self.stat_entries.set(str(entries))
        self.stat_plates.set(str(plates))
        self.stat_active.set(str(active))
        self.stat_total.set(str(total))

        self.root.after(45, self._refresh)

    def on_close(self):
        self.backend.running = False
        self.root.destroy()


# ═══════════════════════════════════════════════════════════════════
#  ENTRY
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Kamera mənbəyi: komanda sətrindən ötürülə bilər
    #   python qaraj_sistemi.py "rtsp://kullaniciadi:sifre@kamera-ip:port/yol"
    #   python qaraj_sistemi.py 0          (yerli veb-kamera)
    # Heç nə verilməsə, lokal veb-kameradan (indeks 0) oxunur.
    if len(sys.argv) > 1:
        src = sys.argv[1]
        if src.isdigit():
            src = int(src)
    else:
        src = 0

    root = Tk()
    app = GarageGUI(root, src)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()