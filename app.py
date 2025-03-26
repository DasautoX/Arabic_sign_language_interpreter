import sys
import time
import cv2
import torch
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display
import mediapipe as mp

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QWidget, QScrollArea, QTabWidget, QStatusBar,
    QSpacerItem, QSizePolicy,QComboBox
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPalette
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QSize

from PIL import Image, ImageDraw, ImageFont
from googletrans import Translator
from edge_tts_util import speak

from helper_functions import (
    PoseLandmarker, HandLandmarker, 
    pose_options, hand_options, frame2npy
)
from MyModel import MyModel

###################################################
# ============ COLOR & LOGO SETTINGS ==============
###################################################
MAROON_COLOR   = "#8D1B3D"
HEADER_COLOR   = "#8D1B3D"  
FOOTER_COLOR   = "#2F2F2F"  # a darker footer
LOGO_PATH      = "assets/qatar_logo.png"  # <-- Replace with actual path

###################################################
# ============== Model Setup ======================
###################################################
model_word = torch.jit.load('assets/model.pt')

ARABIC_TRANSLATIONS = {
    'baby': 'طفل',       'eat': 'يأكل',
    'father': 'أب',      'finish': 'انتهى',
    'good': 'جيد',       'happy': 'سعيد',
    'hear': 'يسمع',      'house': 'منزل',
    'important': 'مهم',  'love': 'حب',
    'mall': 'مول',       'me': 'أنا',
    'mosque': 'مسجد',    'mother': 'أم',
    'normal': 'عادي',    'sad': 'حزين',
    'stop': 'توقف',      'thanks': 'شكراً',
    'thinking': 'يفكر',  'worry': 'قلق',
    'None': 'لا شيء'
}

LIVE_THRESHOLD   = 10
RESET_THRESHOLD  = 5
potential_label  = "None"
current_label    = "None"
live             = 0
no_sign_counter  = 0

try:
    model_char = MyModel()
except Exception as e:
    print("Error loading character model:", e)
    model_char = None

label2text = {
    'aleff': 'أ', 'zay': 'ز', 'seen': 'س', 'sheen': 'ش',
    'saad': 'ص', 'dhad': 'ض', 'taa': 'ط', 'dha': 'ظ',
    'ain': 'ع', 'ghain': 'غ', 'fa': 'ف', 'bb': 'ب',
    'gaaf': 'ق', 'kaaf': 'ك', 'laam': 'ل', 'meem': 'م',
    'nun': 'ن', 'ha': 'ه', 'waw': 'و', 'yaa': 'ئ',
    'toot': 'ة', 'al': 'لا', 'ta': 'ت', 'la': 'ال',
    'ya': 'ى', 'thaa': 'ث', 'jeem': 'ج', 'haa': 'ح',
    'khaa': 'خ', 'dal': 'د', 'thal': 'ذ', 'ra': 'ر'
}

translator = Translator()

last_sign_time      = time.time()
last_detected_label = None

###################################################
# ========= Drawing / Utility Helpers ============
###################################################
def putArTextTop(image, text, color=(61, 27, 141), font_size=50,
                 fontpath='assets/NotoSansArabic.ttf'):
    font = ImageFont.truetype(fontpath, font_size)
    img_pil = Image.fromarray(image)
    draw    = ImageDraw.Draw(img_pil)

    top_bar_height = 70
    top_bar_color  = (245, 245, 245)
    draw.rectangle([(0, 0), (image.shape[1], top_bar_height)],
                   fill=top_bar_color)

    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text     = get_display(reshaped_text)

    bbox        = draw.textbbox((0, 0), bidi_text, font=font)
    text_width  = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    text_x = image.shape[1] - text_width - 20
    text_y = (top_bar_height - 40 - text_height) // 2

    draw.text((text_x, text_y), bidi_text, fill=color, font=font)
    return np.array(img_pil)

def putArTextBottom(image, text, color=(61, 27, 141), font_size=50,
                    fontpath='assets/NotoSansArabic.ttf'):
    font = ImageFont.truetype(fontpath, font_size)
    img_pil = Image.fromarray(image)
    draw    = ImageDraw.Draw(img_pil)

    bottom_bar_height = 70
    bottom_bar_color  = (245, 245, 245)
    bar_start         = image.shape[0] - bottom_bar_height
    draw.rectangle([(0, bar_start), (image.shape[1], image.shape[0])],
                   fill=bottom_bar_color)

    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text     = get_display(reshaped_text)

    bbox        = draw.textbbox((0, 0), bidi_text, font=font)
    text_width  = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    text_x = image.shape[1] - text_width - 20
    text_y = bar_start + (bottom_bar_height - 40 - text_height) // 2

    draw.text((text_x, text_y), bidi_text, fill=color, font=font)
    return np.array(img_pil)

###################################################
# =========== Mediapipe Face Setup ===============
###################################################
mp_face_mesh       = mp.solutions.face_mesh
mp_drawing         = mp.solutions.drawing_utils
mp_drawing_styles  = mp.solutions.drawing_styles

def draw_face_mesh(frame_bgr, face_landmarks):
    mp_drawing.draw_landmarks(
        image=frame_bgr,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
    )

###################################################
# =========== Head/Eyebrow Helpers ===============
###################################################
def get_head_angles(landmarks):
    nose_tip = landmarks[1]
    chin     = landmarks[152]
    pitch    = (nose_tip.y - chin.y) * 1000
    yaw      = (nose_tip.x - 0.5) * 1000
    return pitch, yaw

def get_eye_aspect_ratio(landmarks, indices):
    x_coords = [landmarks[i].x for i in indices]
    y_coords = [landmarks[i].y for i in indices]
    width  = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)
    return width / height if height else 1

###################################################
# ======== FaceGestureDetector (nod, blink) =======
###################################################
class FaceGestureDetector:
    def __init__(self, buffer_size=5):
        self.buffer_size = buffer_size
        self.pitch_buffer = []
        self.yaw_buffer   = []
        self.left_ear_buffer  = []
        self.right_ear_buffer = []

    def update(self, face_landmarks):
        if face_landmarks is None:
            self.pitch_buffer.clear()
            self.yaw_buffer.clear()
            self.left_ear_buffer.clear()
            self.right_ear_buffer.clear()
            return None

        left_eye_idx  = [33, 160, 158, 133, 153, 144]
        right_eye_idx = [362, 385, 387, 263, 373, 380]

        left_ear  = get_eye_aspect_ratio(face_landmarks, left_eye_idx)
        right_ear = get_eye_aspect_ratio(face_landmarks, right_eye_idx)
        pitch, yaw = get_head_angles(face_landmarks)

        self.pitch_buffer.append(pitch)
        self.yaw_buffer.append(yaw)
        self.left_ear_buffer.append(left_ear)
        self.right_ear_buffer.append(right_ear)

        if len(self.pitch_buffer) > self.buffer_size:
            self.pitch_buffer.pop(0)
            self.yaw_buffer.pop(0)
            self.left_ear_buffer.pop(0)
            self.right_ear_buffer.pop(0)

        if len(self.pitch_buffer) == self.buffer_size:
            return self.detect_gesture()
        return None

    def detect_gesture(self):
        half = self.buffer_size // 2
        old_pitch_avg = np.mean(self.pitch_buffer[:half])
        new_pitch_avg = np.mean(self.pitch_buffer[half:])
        pitch_diff    = new_pitch_avg - old_pitch_avg

        old_yaw_avg = np.mean(self.yaw_buffer[:half])
        new_yaw_avg = np.mean(self.yaw_buffer[half:])
        yaw_diff    = new_yaw_avg - old_yaw_avg

        closed_threshold = 0.2
        left_closed_count  = sum([1 for val in self.left_ear_buffer  if val < closed_threshold])
        right_closed_count = sum([1 for val in self.right_ear_buffer if val < closed_threshold])

        if left_closed_count >= half and right_closed_count >= half:
            return "blink"

        if pitch_diff < -10:
            return "nod"

        if yaw_diff > 15:
            return "shake_right"
        if yaw_diff < -15:
            return "shake_left"

        return None

###################################################
# ========== Eyebrow Raise Detection =============
###################################################
eyebrows_open_time = 0.0
last_eyebrow_time  = time.time()

def detect_eyebrows_raised_time_based(face_landmarks, dt, thr=0.08, hold_time=0.5):
    global eyebrows_open_time

    left_eyebrow_y  = (face_landmarks[70].y + face_landmarks[63].y) / 2
    right_eyebrow_y = (face_landmarks[300].y + face_landmarks[296].y) / 2
    left_eye_y      = face_landmarks[145].y
    right_eye_y     = face_landmarks[374].y

    dist_left  = left_eye_y  - left_eyebrow_y
    dist_right = right_eye_y - right_eyebrow_y

    if dist_left > thr and dist_right > thr:
        eyebrows_open_time += dt
    else:
        eyebrows_open_time = 0.0

    if eyebrows_open_time > hold_time:
        eyebrows_open_time = 0.0
        return True
    return False

###################################################
# ================= HEAD TILTS ===================
###################################################
tilt_left_open_time  = 0.0
tilt_right_open_time = 0.0
last_tilt_left_time  = time.time()
last_tilt_right_time = time.time()

def detect_head_tilt_left_time_based(roll_diff, dt, tilt_thr=0.05, hold_time=0.5):
    global tilt_left_open_time
    if roll_diff > tilt_thr:
        tilt_left_open_time += dt
    else:
        tilt_left_open_time = 0.0

    if tilt_left_open_time > hold_time:
        tilt_left_open_time = 0.0
        return True
    return False

def detect_head_tilt_right_time_based(roll_diff, dt, tilt_thr=0.05, hold_time=0.5):
    global tilt_right_open_time
    if roll_diff < -tilt_thr:
        tilt_right_open_time += dt
    else:
        tilt_right_open_time = 0.0

    if tilt_right_open_time > hold_time:
        tilt_right_open_time = 0.0
        return True
    return False

###################################################
# ================ CaptureThread =================
###################################################
class CaptureThread(QThread):
    frame_ready     = pyqtSignal(np.ndarray)
    text_ar_updated = pyqtSignal(str)
    text_en_updated = pyqtSignal(str)
    eyebrow_signal  = pyqtSignal(bool)
    mode_changed    = pyqtSignal(str)
    status_message  = pyqtSignal(str)  # for status bar updates

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running    = True
        self.mode       = "word"
        self.final_text_ar = ""
        self.final_text_en = ""
        self.draw_face_mesh_flag = True  # toggle to show/hide face mesh
        self.gesture_control = {
            "nod": True,
            "tilt_left": True,
            "tilt_right": True,
            "eyebrow_raise": True,
            }

        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.gesture_detector  = FaceGestureDetector(buffer_size=5)
        self.last_gesture_time = 0.0
        self.gesture_cooldown  = 2.0

        global potential_label, current_label, live, no_sign_counter
        potential_label = "None"
        current_label   = "None"
        live            = 0
        no_sign_counter = 0

        global last_sign_time, last_detected_label
        last_sign_time      = time.time()
        last_detected_label = None

        self.poselandmarker = PoseLandmarker.create_from_options(pose_options)
        self.handlandmarker = HandLandmarker.create_from_options(hand_options)

    def run(self):
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            frame_timestamp_ms = int(self.capture.get(cv2.CAP_PROP_POS_MSEC))
            frame_npy = frame2npy(frame, frame_timestamp_ms,
                                  self.poselandmarker, self.handlandmarker)

            # Word or Char mode logic
            if self.mode == "word":
                frame_npy = np.expand_dims(frame_npy, (0, 1))
                frame_t   = torch.from_numpy(frame_npy).float().to(model_word.device)
                label     = model_word.predict(frame_t)['labels'][0]

                global no_sign_counter, current_label, potential_label, live
                if label == "None":
                    no_sign_counter += 1
                    if no_sign_counter >= RESET_THRESHOLD:
                        current_label   = "None"
                        potential_label = "None"
                        live            = 0
                else:
                    no_sign_counter = 0
                    if label == potential_label:
                        live += 1
                    else:
                        potential_label = label
                        live = 0

                if live > LIVE_THRESHOLD:
                    current_label = potential_label

                arabic_word = ARABIC_TRANSLATIONS[current_label]
                frame = putArTextTop(frame, arabic_word)
            else:
                global model_char, last_sign_time, last_detected_label
                if model_char is not None:
                    try:
                        label_list, prop_list = model_char.predict([frame])
                        label_char = label_list[0]
                        prob_char  = prop_list[0]
                    except:
                        label_char = '-1'
                        prob_char  = 0.0

                    now = time.time()
                    if label_char != '-1' and prob_char > 0.5:
                        if label_char != last_detected_label:
                            last_sign_time      = now
                            last_detected_label = label_char

                        if (now - last_sign_time) >= 2:
                            arabic_char = label2text.get(label_char, '')
                            self.final_text_ar += arabic_char
                            self.broadcast_texts()
                            last_sign_time = now

                        frame = putArTextTop(frame, label2text.get(label_char, ''))
                    else:
                        frame = putArTextTop(frame, "Please Show a Letter Sign")
                else:
                    frame = putArTextTop(frame, "Character model not loaded!")

            with mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as face_mesh:
                results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                face_landmarks = None
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0].landmark
                    if self.draw_face_mesh_flag:
                        draw_face_mesh(frame, results.multi_face_landmarks[0])

            if face_landmarks:
                left_ear_y  = face_landmarks[454].y
                right_ear_y = face_landmarks[234].y
                roll_diff   = left_ear_y - right_ear_y

                now = time.time()

                global last_tilt_left_time, last_tilt_right_time
                dt_left  = now - last_tilt_left_time
                dt_right = now - last_tilt_right_time

                if detect_head_tilt_left_time_based(roll_diff, dt_left) and ((now - self.last_gesture_time) > self.gesture_cooldown) and self.gesture_control["tilt_left"]:
                    if self.mode == "word" and current_label != "None":
                        self.final_text_ar += ARABIC_TRANSLATIONS[current_label] + " "
                        self.broadcast_texts()
                        self.status_message.emit("Tilt left => Added recognized word")
                    self.last_gesture_time = now

                if detect_head_tilt_right_time_based(roll_diff, dt_right) and ((now - self.last_gesture_time) > self.gesture_cooldown) and self.gesture_control["tilt_right"]:
                    if self.mode == "word":
                        speak(self.final_text_ar)
                        self.status_message.emit("Tilt right => Spoke final text (AR)")
                    elif self.mode == "char":
                        self.final_text_ar += " "
                        self.broadcast_texts()
                        self.status_message.emit("Tilt right => Added a space (Char Mode)")
                    self.last_gesture_time = now

                last_tilt_left_time  = now
                last_tilt_right_time = now

                global last_eyebrow_time
                dt_eb = now - last_eyebrow_time
                eyebrow_raised = detect_eyebrows_raised_time_based(face_landmarks, dt_eb)
                if eyebrow_raised and ((now - self.last_gesture_time) > self.gesture_cooldown) and self.gesture_control["eyebrow_raise"]:
                    self.eyebrow_signal.emit(True)
                    if self.final_text_ar.strip():
                        translated         = translator.translate(self.final_text_ar, src="ar", dest="en")
                        self.final_text_en = translated.text
                        speak(self.final_text_en)
                        self.status_message.emit("Eyebrows => Translated AR->EN & Spoke!")
                    else:
                        if self.final_text_en.strip():
                            translated         = translator.translate(self.final_text_en, src="en", dest="ar")
                            self.final_text_ar = translated.text
                            speak(self.final_text_ar)
                            self.status_message.emit("Eyebrows => Translated EN->AR & Spoke!")

                    self.broadcast_texts()
                    self.last_gesture_time = now
                else:
                    self.eyebrow_signal.emit(False)

                last_eyebrow_time = now

                gesture = self.gesture_detector.update(face_landmarks)
                if gesture and (now - self.last_gesture_time) > self.gesture_cooldown:
                    if gesture == "nod" and self.gesture_control["nod"]:
                        self.toggle_mode()
                        self.status_message.emit("Nod => Toggled mode!")
                    elif gesture in ("shake_left", "shake_right"):
                        if self.mode == "char" and self.final_text_ar:
                            self.final_text_ar = self.final_text_ar[:-1]
                            self.broadcast_texts()
                            self.status_message.emit("Shake => Removed last char (char mode).")
                    elif gesture == "blink":
                        pass
                    self.last_gesture_time = now

            frame_mesh = putArTextBottom(frame, self.final_text_ar)
            self.frame_ready.emit(frame_mesh)

        self.capture.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

    def toggle_mode(self):
        if self.mode == "word":
            self.mode = "char"
        else:
            self.mode = "word"
        self.mode_changed.emit(self.mode)

    def remove_last_char(self):
        if self.final_text_ar:
            self.final_text_ar = self.final_text_ar[:-1]
            self.broadcast_texts()

    def speak_ar(self):
        speak(self.final_text_ar)
        self.status_message.emit("Spoke Arabic text.")

    def speak_en(self):
        speak(self.final_text_en)
        self.status_message.emit("Spoke English text.")

    def reset_texts(self):
        self.final_text_ar = ""
        self.final_text_en = ""
        self.broadcast_texts()
        self.status_message.emit("Reset all text.")

    def toggle_face_mesh(self):
        self.draw_face_mesh_flag = not self.draw_face_mesh_flag
        if self.draw_face_mesh_flag:
            self.status_message.emit("Face Mesh: ON")
        else:
            self.status_message.emit("Face Mesh: OFF")

    def broadcast_texts(self):
        if self.final_text_ar.strip():
            translated = translator.translate(self.final_text_ar, src="ar", dest="en")
            self.final_text_en = translated.text
        else:
            self.final_text_en = ""

        self.text_ar_updated.emit(self.final_text_ar)
        self.text_en_updated.emit(self.final_text_en)

###################################################
# ================ MainWindow =====================
###################################################
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Interpreter")
        self.resize(1300, 850)

        # Track current language
        self.current_language = "English"

        # We'll add a status bar at the bottom
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(f"background-color: {FOOTER_COLOR}; color: white;")
        self.setStatusBar(self.status_bar)

        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: #FFFFFF;  /* White background */
            }}
            QPushButton {{
                background-color: #FFFFFF;
                color: {MAROON_COLOR};
                font-size: 16px;
                font-weight: bold;
                border: 2px solid {MAROON_COLOR};
                border-radius: 8px;
                padding: 6px;
            }}
            QPushButton:hover {{
                background-color: #f5e6ec;
            }}
            QLabel {{
                color: {MAROON_COLOR};
                font-size: 18px;
                font-weight: bold;
            }}
        """)

        # main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # ========== Header Bar ==========
        header_bar = QWidget()
        header_bar.setFixedHeight(50)
        header_bar.setStyleSheet(f"background-color: {HEADER_COLOR};")

        header_layout = QHBoxLayout(header_bar)
        header_layout.setContentsMargins(10, 5, 10, 5)

        # App title label
        self.label_app_title = QLabel()
        self.label_app_title.setStyleSheet("color: white; font-size: 20px; font-weight: bold;")
        header_layout.addWidget(self.label_app_title)

        header_layout.addStretch()

        # Language ComboBox
        self.language_combo = QComboBox()
        self.language_combo.addItem("English")
        self.language_combo.addItem("Arabic")
        self.language_combo.currentIndexChanged.connect(self.change_language)
        header_layout.addWidget(self.language_combo)

        # face mesh toggle button
        self.btn_toggle_mesh = QPushButton()
        self.btn_toggle_mesh.setToolTip("Show or hide the face mesh lines on camera feed.")
        self.btn_toggle_mesh.setStyleSheet("background-color: white; color: #8D1B3D;")
        self.btn_toggle_mesh.clicked.connect(self.toggle_face_mesh_in_thread)
        header_layout.addWidget(self.btn_toggle_mesh)

        main_layout.addWidget(header_bar)

        # ========== Central Split Layout ==========
        split_layout = QHBoxLayout()

        # ---------- Left Panel: TABS (logo, instructions, advanced) ----------
        left_panel = QVBoxLayout()

        self.logo_label = QLabel()
        try:
            pixmap_logo = QPixmap(LOGO_PATH)
            pixmap_logo = pixmap_logo.scaled(180, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.logo_label.setPixmap(pixmap_logo)
        except:
            self.logo_label.setText("LOGO")
        left_panel.addWidget(self.logo_label, alignment=Qt.AlignTop)

        tab_widget = QTabWidget()
        tab_widget.setStyleSheet(f"QTabBar::tab {{ font-size: 14px; color: {MAROON_COLOR}; }}")

        # Secondary logo layout
        logo_layout = QHBoxLayout()
        self.secondary_logo_label = QLabel()
        try:
            secondary_pixmap_logo = QPixmap("assets/comp_logo.png")  # Replace with actual path
            secondary_pixmap_logo = secondary_pixmap_logo.scaled(180, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.secondary_logo_label.setPixmap(secondary_pixmap_logo)
        except:
            self.secondary_logo_label.setText("LOGO 2")
        logo_layout.addWidget(self.secondary_logo_label, alignment=Qt.AlignRight)
        left_panel.addLayout(logo_layout)

        # Instructions tab
        instructions_tab = QWidget()
        instructions_layout = QVBoxLayout(instructions_tab)
        instructions_scroll = QScrollArea()
        instructions_scroll.setWidgetResizable(True)

        instructions_widget = QWidget()
        instructions_widget_layout = QVBoxLayout(instructions_widget)
        self.label_instructions = QLabel()
        self.label_instructions.setStyleSheet("font-size: 14px;")
        instructions_widget_layout.addWidget(self.label_instructions)
        instructions_widget_layout.addStretch()

        instructions_scroll.setWidget(instructions_widget)
        instructions_layout.addWidget(instructions_scroll)
        tab_widget.addTab(instructions_tab, "")

        # Advanced tab
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout(advanced_tab)
        self.label_advanced = QLabel()
        self.label_advanced.setStyleSheet("font-size: 14px;")
        advanced_layout.addWidget(self.label_advanced)
        advanced_layout.addStretch()
        tab_widget.addTab(advanced_tab, "")

        left_panel.addWidget(tab_widget)
        left_panel.addStretch()

        split_layout.addLayout(left_panel, 35)

        # Create separate widget for gesture controls and put it in the instructions layout
        self.gesture_buttons = {}
        gesture_control_widget = QWidget()
        gesture_control_layout = QVBoxLayout(gesture_control_widget)

        self.gesture_label = QLabel()
        self.gesture_label.setStyleSheet("font-size: 16px; color: #8D1B3D;")
        gesture_control_layout.addWidget(self.gesture_label)

        gestures = ["Nod", "Tilt Left", "Tilt Right", "Eyebrow Raise"]
        for gesture in gestures:
            btn = QPushButton()
            btn.setCheckable(True)
            btn.setChecked(True)  # Default: enabled
            # Use a lambda to capture the gesture name in lower/underscored form
            btn.clicked.connect(
                lambda checked, g=gesture.lower().replace(" ", "_"): self.toggle_gesture(g, checked)
            )
            gesture_control_layout.addWidget(btn)
            self.gesture_buttons[gesture] = btn

        gesture_control_layout.addStretch()
        instructions_layout.addWidget(gesture_control_widget)

        # ---------- Right Panel (camera, buttons, text) ----------
        right_panel = QVBoxLayout()

        self.label_mode = QLabel()
        self.label_mode.setToolTip("Displays whether you are in 'word' or 'character' mode.")
        right_panel.addWidget(self.label_mode, alignment=Qt.AlignRight)

        self.camera_label = QLabel()
        self.camera_label.setFixedSize(QSize(960, 540))
        self.camera_label.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(self.camera_label)

        # Button row
        btn_layout = QHBoxLayout()

        self.btn_toggle_mode = QPushButton()
        self.btn_toggle_mode.setToolTip("Switch between word mode or character mode.")
        self.btn_toggle_mode.clicked.connect(self.toggle_mode)
        btn_layout.addWidget(self.btn_toggle_mode)

        self.btn_delete_char = QPushButton()
        self.btn_delete_char.setToolTip("Remove the last recognized character in Char mode.")
        self.btn_delete_char.clicked.connect(self.delete_char)
        btn_layout.addWidget(self.btn_delete_char)

        self.btn_speak_ar = QPushButton()
        self.btn_speak_ar.setToolTip("Speaks the Arabic text aloud.")
        self.btn_speak_ar.clicked.connect(self.speak_ar)
        btn_layout.addWidget(self.btn_speak_ar)

        self.btn_speak_en = QPushButton()
        self.btn_speak_en.setToolTip("Speaks the English text aloud.")
        self.btn_speak_en.clicked.connect(self.speak_en)
        btn_layout.addWidget(self.btn_speak_en)

        self.btn_reset = QPushButton()
        self.btn_reset.setToolTip("Clears the final text in both AR and EN.")
        self.btn_reset.clicked.connect(self.reset_texts)
        btn_layout.addWidget(self.btn_reset)

        right_panel.addLayout(btn_layout)

        self.label_ar = QLabel()
        right_panel.addWidget(self.label_ar)

        self.label_en = QLabel()
        right_panel.addWidget(self.label_en)

        self.label_eyebrow = QLabel()
        right_panel.addWidget(self.label_eyebrow)

        right_panel.addStretch()
        split_layout.addLayout(right_panel, 65)

        main_layout.addLayout(split_layout)

        # ========== Footer Bar ==========
        footer_bar = QWidget()
        footer_bar.setFixedHeight(30)
        footer_bar.setStyleSheet(f"background-color: {FOOTER_COLOR};")
        footer_layout = QHBoxLayout(footer_bar)
        footer_layout.setContentsMargins(5, 0, 5, 0)

        self.footer_label = QLabel()
        self.footer_label.setStyleSheet("color: white; font-size: 12px;")
        footer_layout.addWidget(self.footer_label, alignment=Qt.AlignLeft)

        # push right
        footer_layout.addStretch()
        main_layout.addWidget(footer_bar)

        # ========== Start Capture Thread ==========
        self.capture_thread = CaptureThread()
        self.capture_thread.frame_ready.connect(self.update_frame)
        self.capture_thread.text_ar_updated.connect(self.update_text_ar)
        self.capture_thread.text_en_updated.connect(self.update_text_en)
        self.capture_thread.eyebrow_signal.connect(self.update_eyebrow_label)
        self.capture_thread.mode_changed.connect(self.update_mode_label)
        self.capture_thread.status_message.connect(self.show_status_message)
        self.capture_thread.start()

        # Initialize UI text for the chosen language (default "English")
        self.change_language()

    # ========== Language Toggle ==========
    def change_language(self):
        """Update the entire UI to the selected language."""
        self.current_language = self.language_combo.currentText()
        tab_widget = self.findChild(QTabWidget)

        if self.current_language == "Arabic":
            # Window and Title
            self.setWindowTitle("مترجم لغة الإشارة")
            self.label_app_title.setText("مترجم لغة الإشارة")

            # Face Mesh Toggle
            self.btn_toggle_mesh.setText("إظهار/إخفاء خطوط الشبكية")

            # Tab Titles
            if tab_widget:
                tab_widget.setTabText(0, "تعليمات")
                tab_widget.setTabText(1, "متقدم")

            # Instructions
            self.label_instructions.setText(
                "<b>تعليمات الحركات:</b><br>"
                "• إمالة إلى اليسار => إضافة الكلمة المكتشفة (وضع الكلمات)<br>"
                "• إمالة إلى اليمين => نطق النص النهائي<br>"
                "• رفع الحواجب => الترجمة بين العربية والإنجليزية<br>"
                "• إيماءة بالرأس => تبديل وضع الكلمات/الأحرف<br>"
                "• هز بالرأس => إزالة آخر حرف (وضع الأحرف)<br>"
                "• رمشة => غير مخصص<br><br>"
                "<b>وظائف إضافية:</b><br>"
                "• تكلم عربي / تكلم إنجليزي => قراءة النصوص بصوت عالٍ<br>"
                "• إعادة التعيين => مسح النصوص بالعربية والإنجليزية<br>"
                "• إظهار/إخفاء خطوط الشبكية => إظهار أو إخفاء خطوط Mediapipe<br>"
                "<br><i>استمتع بالتجربة!</i>"
            )

            # Advanced Label
            self.label_advanced.setText(
                "<b>الإعدادات المتقدمة:</b><br>"
                "قريباً..."
            )

            # Gesture Control label
            self.gesture_label.setText("<b>التحكم بالحركات:</b>")

            # Gesture Buttons
            self.gesture_buttons["Nod"].setText("تفعيل إيماءة الرأس")
            self.gesture_buttons["Tilt Left"].setText("تفعيل إمالة اليسار")
            self.gesture_buttons["Tilt Right"].setText("تفعيل إمالة اليمين")
            self.gesture_buttons["Eyebrow Raise"].setText("تفعيل رفع الحواجب")

            # Right panel
            self.label_mode.setText("الوضع الحالي: كلمة")
            self.btn_toggle_mode.setText("تبديل الوضع (كلمة/حرف)")
            self.btn_delete_char.setText("حذف آخر حرف")
            self.btn_speak_ar.setText("تكلم عربي")
            self.btn_speak_en.setText("تكلم إنجليزي")
            self.btn_reset.setText("إعادة التعيين")
            self.label_ar.setText("النص النهائي (عربي):")
            self.label_en.setText("النص النهائي (إنجليزي):")
            self.label_eyebrow.setText("الحواجب: غير مكتشفة")

            # Footer
            self.footer_label.setText("© 2023 - مترجم لغة الإشارة المطور")
            self.status_bar.showMessage("تم التبديل إلى اللغة العربية", 3000)

        else:
            # English UI
            self.setWindowTitle("Sign Language Interpreter")
            self.label_app_title.setText("Sign Language Interpreter")

            # Face Mesh Toggle
            self.btn_toggle_mesh.setText("Toggle Face Mesh")

            # Tab Titles
            if tab_widget:
                tab_widget.setTabText(0, "Instructions")
                tab_widget.setTabText(1, "Advanced")

            # Instructions
            self.label_instructions.setText(
                "<b>Gesture Instructions:</b><br>"
                "• Tilt Left => Add recognized word (Word Mode)<br>"
                "• Tilt Right => Speak Final Text<br>"
                "• Raise Eyebrows => Translate AR <-> EN<br>"
                "• Nod => Toggle Word/Char Mode<br>"
                "• Shake => Remove last char (Char Mode)<br>"
                "• Blink => Not Assigned<br><br>"
                "<b>Extra Functions:</b><br>"
                "• Speak AR / Speak EN => read texts aloud<br>"
                "• Reset All => clears AR & EN text<br>"
                "• Toggle Face Mesh => show/hide Mediapipe lines<br>"
                "<br><i>Have fun exploring!</i>"
            )

            # Advanced Label
            self.label_advanced.setText(
                "<b>Advanced Settings:</b><br>"
                "Coming soon..."
            )

            # Gesture Control label
            self.gesture_label.setText("<b>Gesture Control:</b>")

            # Gesture Buttons
            self.gesture_buttons["Nod"].setText("Enable Nod")
            self.gesture_buttons["Tilt Left"].setText("Enable Tilt Left")
            self.gesture_buttons["Tilt Right"].setText("Enable Tilt Right")
            self.gesture_buttons["Eyebrow Raise"].setText("Enable Eyebrow Raise")

            # Right panel
            self.label_mode.setText("Current Mode: Word")
            self.btn_toggle_mode.setText("Toggle Mode (Word/Char)")
            self.btn_delete_char.setText("Delete Last Char")
            self.btn_speak_ar.setText("Speak AR")
            self.btn_speak_en.setText("Speak EN")
            self.btn_reset.setText("Reset All")
            self.label_ar.setText("Final Text (AR):")
            self.label_en.setText("Final Text (EN):")
            self.label_eyebrow.setText("Eyebrows: Not Detected")

            # Footer
            self.footer_label.setText("© 2023 - Enhanced Sign Language Interpreter")
            self.status_bar.showMessage("Switched to English", 3000)

    # ============= Slots ==============
    def update_frame(self, frame_bgr: np.ndarray):
        h, w, ch = frame_bgr.shape
        bytes_per_line = ch * w
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.camera_label.setPixmap(
            pix.scaled(self.camera_label.width(), self.camera_label.height(), Qt.KeepAspectRatio)
        )

    def toggle_gesture(self, gesture: str, enabled: bool):
        """
        Enable or disable a specific gesture.
        """
        if hasattr(self.capture_thread, 'gesture_control'):
            self.capture_thread.gesture_control[gesture] = enabled  # Update gesture state in CaptureThread
            state = "enabled" if enabled else "disabled"
            self.show_status_message(f"{gesture.replace('_', ' ').capitalize()} gesture {state}.")
        else:
            self.show_status_message("Gesture control is not initialized.")

    def update_text_ar(self, text_ar: str):
        # Update AR text according to language
        if self.current_language == "Arabic":
            self.label_ar.setText("النص النهائي (عربي): " + text_ar)
        else:
            self.label_ar.setText("Final Text (AR): " + text_ar)

    def update_text_en(self, text_en: str):
        # Update EN text according to language
        if self.current_language == "Arabic":
            self.label_en.setText("النص النهائي (إنجليزي): " + text_en)
        else:
            self.label_en.setText("Final Text (EN): " + text_en)

    def update_eyebrow_label(self, is_raised: bool):
        # Update eyebrow label text
        if self.current_language == "Arabic":
            self.label_eyebrow.setText("الحواجب: مكتشفة" if is_raised else "الحواجب: غير مكتشفة")
        else:
            self.label_eyebrow.setText("Eyebrows: DETECTED" if is_raised else "Eyebrows: Not Detected")

    def update_mode_label(self, mode_str: str):
        # Update the mode label to the current language
        if self.current_language == "Arabic":
            if mode_str.lower() == "word":
                self.label_mode.setText("الوضع الحالي: كلمة")
            else:
                self.label_mode.setText("الوضع الحالي: حرف")
        else:
            self.label_mode.setText(f"Current Mode: {mode_str.capitalize()}")

    def show_status_message(self, msg: str):
        """Show a status message in the status bar."""
        self.status_bar.showMessage(msg, 3000)  # 3-second message

    # ========== Buttons ==========
    def toggle_mode(self):
        self.capture_thread.toggle_mode()

    def delete_char(self):
        self.capture_thread.remove_last_char()

    def speak_ar(self):
        self.capture_thread.speak_ar()

    def speak_en(self):
        self.capture_thread.speak_en()

    def reset_texts(self):
        self.capture_thread.reset_texts()

    def toggle_face_mesh_in_thread(self):
        self.capture_thread.toggle_face_mesh()

    def closeEvent(self, event):
        self.capture_thread.stop()
        event.accept()

###################################################
# ================== main() =======================
###################################################
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
