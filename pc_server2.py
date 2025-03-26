import sys
import time
import cv2
import torch
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display
import mediapipe as mp
from flask import Flask, request, jsonify, Response
from PIL import Image, ImageDraw, ImageFont
import base64

from googletrans import Translator

from helper_functions import (
    PoseLandmarker, HandLandmarker,
    pose_options, hand_options, frame2npy
)
from MyModel import MyModel

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

###################################################
# ============== Model Setup ======================
###################################################
model_word = torch.jit.load('assets/model.pt').to(device) # Move model to GPU
model_word.eval()  # Set the model to evaluation mode

ARABIC_TRANSLATIONS = {
    'baby': 'طفل',     'eat': 'يأكل',
    'father': 'أب',      'finish': 'انتهى',
    'good': 'جيد',      'happy': 'سعيد',
    'hear': 'يسمع',     'house': 'منزل',
    'important': 'مهم',  'love': 'حب',
    'mall': 'مول',      'me': 'أنا',
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

    bbox       = draw.textbbox((0, 0), bidi_text, font=font)
    text_width = bbox[2] - bbox[0]
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

    bbox       = draw.textbbox((0, 0), bidi_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    text_x = image.shape[1] - text_width - 20
    text_y = bar_start + (bottom_bar_height - 40 - text_height) // 2

    draw.text((text_x, text_y), bidi_text, fill=color, font=font)
    return np.array(img_pil)

###################################################
# =========== Mediapipe Face Setup ===============
###################################################
mp_face_mesh      = mp.solutions.face_mesh
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

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
# Define these variables in the global scope 
tilt_left_open_time  = 0.0
tilt_right_open_time = 0.0
last_tilt_left_time  = time.time()
last_tilt_right_time = time.time()

def detect_head_tilt_left_time_based(roll_diff, dt, tilt_thr=0.05, hold_time=0.5):
    global tilt_left_open_time, last_tilt_left_time # Declare as global
    if roll_diff > tilt_thr:
        tilt_left_open_time += dt
    else:
        tilt_left_open_time = 0.0

    if tilt_left_open_time > hold_time:
        tilt_left_open_time = 0.0
        return True
    return False

def detect_head_tilt_right_time_based(roll_diff, dt, tilt_thr=0.05, hold_time=0.5):
    global tilt_right_open_time, last_tilt_right_time # Declare as global
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
class CaptureThread:

    def __init__(self):
        self.running             = True
        self.mode                = "word"
        self.final_text_ar       = ""
        self.final_text_en       = ""
        self.draw_face_mesh_flag = True  # toggle to show/hide face mesh
        self.gesture_control     = {
            "nod": True,
            "tilt_left": True,
            "tilt_right": True,
            "eyebrow_raise": True,
            "shake": True
            }

        # Note: No video capture on the server

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

    def process_frame(self, frame):
        """
        Processes a single frame (replace run() logic).
        """
        # Ensure that necessary global variables are accessible
        global last_tilt_left_time, last_tilt_right_time, last_eyebrow_time
        frame = cv2.flip(frame, 1)
        frame_timestamp_ms = int(time.time() * 1000)  # Use time.time() for timestamp
        frame_npy = frame2npy(frame, frame_timestamp_ms,
                              self.poselandmarker, self.handlandmarker)

        # Word or Char mode logic
        if self.mode == "word":
            frame_npy = np.expand_dims(frame_npy, (0, 1))
            frame_t = torch.from_numpy(frame_npy).float().to(device)
            with torch.no_grad():  # Disable gradient calculations
                label = model_word.predict(frame_t)['labels'][0]

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

        else:  # Char mode
            global model_char, last_sign_time, last_detected_label
            if model_char is not None:
                try:
                    
                    label_list, prop_list = model_char.predict([frame])
                    label_char = label_list[0]
                    prob_char  = prop_list[0]
                except Exception as e:
                    print(f"Error in character prediction: {e}")
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
                        last_sign_time = now

                frame = putArTextTop(frame, label2text.get(label_char, ''))

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
            
            # Gesture logic
            if detect_head_tilt_left_time_based(roll_diff, now - last_tilt_left_time) and ((now - self.last_gesture_time) > self.gesture_cooldown) and self.gesture_control["tilt_left"]:
                if self.mode == "word" and current_label != "None":
                    self.final_text_ar += ARABIC_TRANSLATIONS[current_label] + " "
                    self.last_gesture_time = now
                last_tilt_left_time = now # Update the time inside the if

            if detect_head_tilt_right_time_based(roll_diff, now - last_tilt_right_time) and ((now - self.last_gesture_time) > self.gesture_cooldown) and self.gesture_control["tilt_right"]:
                if self.mode == "word":
                    # We will now send a request to the client to speak
                    pass
                elif self.mode == "char":
                    self.final_text_ar += " "
                self.last_gesture_time = now
                last_tilt_right_time = now # Update the time inside the if

            
            eyebrow_raised = detect_eyebrows_raised_time_based(face_landmarks, now - last_eyebrow_time)
            if eyebrow_raised and ((now - self.last_gesture_time) > self.gesture_cooldown) and self.gesture_control["eyebrow_raise"]:
                if self.final_text_ar.strip():
                    translated = translator.translate(self.final_text_ar, src="ar", dest="en")
                    self.final_text_en = translated.text
                else:
                    if self.final_text_en.strip():
                        translated = translator.translate(self.final_text_en, src="en", dest="ar")
                        self.final_text_ar = translated.text
                self.last_gesture_time = now
            last_eyebrow_time = now # Update the time inside the if

            gesture = self.gesture_detector.update(face_landmarks)
            if gesture and (now - self.last_gesture_time) > self.gesture_cooldown:
                if gesture == "nod" and self.gesture_control["nod"]:
                    self.toggle_mode()
                elif gesture in ("shake_left", "shake_right") and self.gesture_control["shake"]:
                    if self.mode == "char" and self.final_text_ar:
                        self.final_text_ar = self.final_text_ar[:-1]
                elif gesture == "blink":
                    pass
                self.last_gesture_time = now

        frame_display = putArTextBottom(frame, self.final_text_ar)

        # Convert the processed frame to JPEG
        _, encoded_frame = cv2.imencode('.jpg', frame_display)
        processed_frame_bytes = encoded_frame.tobytes()

        return {
            "mode": self.mode,
            "final_text_ar": self.final_text_ar,
            "final_text_en": self.final_text_en,
            "current_label": current_label,
            "draw_face_mesh": self.draw_face_mesh_flag,
            "gesture_control": self.gesture_control,
            "processed_frame": processed_frame_bytes  # Add processed frame
        }

    def stop(self):
        self.running = False

    def toggle_mode(self):
        if self.mode == "word":
            self.mode = "char"
        else:
            self.mode = "word"

    def remove_last_char(self):
        if self.final_text_ar:
            self.final_text_ar = self.final_text_ar[:-1]

    def reset_texts(self):
        self.final_text_ar = ""
        self.final_text_en = ""

    def toggle_face_mesh(self):
        self.draw_face_mesh_flag = not self.draw_face_mesh_flag

    def update_gesture_control(self, gesture, enabled):
        self.gesture_control[gesture] = enabled

# Global instance of the processing thread (renamed from CaptureThread)
processing_thread = CaptureThread()

###################################################
# ================= Flask App ====================
###################################################
app = Flask(__name__)

@app.route('/start', methods=['POST'])
def start_processing():
    # We might not need this on the server since
    # processing starts when a frame is received.
    return jsonify({"message": "Processing started (dummy)"})

@app.route('/stop', methods=['POST'])
def stop_processing():
    processing_thread.stop()
    return jsonify({"message": "Processing stopped"})

@app.route('/toggle_mode', methods=['POST'])
def toggle_mode():
    processing_thread.toggle_mode()
    return jsonify({"message": "Mode toggled", "mode": processing_thread.mode})

@app.route('/remove_last_char', methods=['POST'])
def remove_last_char():
    processing_thread.remove_last_char()
    return jsonify({"message": "Last character removed", "text_ar": processing_thread.final_text_ar})

@app.route('/reset_texts', methods=['POST'])
def reset_texts():
    processing_thread.reset_texts()
    return jsonify({"message": "Texts reset", "text_ar": "", "text_en": ""})

@app.route('/toggle_face_mesh', methods=['POST'])
def toggle_face_mesh():
    processing_thread.toggle_face_mesh()
    return jsonify({"message": "Face mesh toggled", "draw_face_mesh": processing_thread.draw_face_mesh_flag})

@app.route('/update_gesture', methods=['POST'])
def update_gesture():
    data = request.get_json()
    gesture = data.get('gesture')
    enabled = data.get('enabled')
    if gesture and enabled is not None:
        processing_thread.update_gesture_control(gesture, enabled)
        return jsonify({"message": f"Gesture control updated: {gesture} = {enabled}"})
    else:
        return jsonify({"error": "Invalid gesture or enabled status"}), 400

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        file = request.files['frame'].read()
        npimg = np.frombuffer(file, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        data = processing_thread.process_frame(frame)

        # Convert the processed frame bytes to base64 for sending in JSON
        data["processed_frame"] = base64.b64encode(data["processed_frame"]).decode('utf-8')

        # Send processed data back to the client
        return jsonify(data)

    except Exception as e:
        print(f"Error in process_frame: {e}")  # Print the error for debugging
        return jsonify({"error": str(e)}), 500
    
@app.route('/get_current_data', methods=['GET'])
def get_current_data():
    """
    Provides an endpoint for the client to get the current state 
    (text ar, text en, mode, etc.) without processing a frame.
    """
    data = {
        "mode": processing_thread.mode,
        "final_text_ar": processing_thread.final_text_ar,
        "final_text_en": processing_thread.final_text_en,
        "current_label": current_label,
        "draw_face_mesh": processing_thread.draw_face_mesh_flag,
        "gesture_control": processing_thread.gesture_control
    }
    return jsonify(data)

# Error handling
@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    # Start the processing thread (not capturing, just processing)
    # processing_thread.run() # You might not need to explicitly start it here

    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)