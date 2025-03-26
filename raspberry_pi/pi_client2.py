import sys
import time
import cv2
import requests
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QWidget, QScrollArea, QTabWidget, QStatusBar,
    QSpacerItem, QSizePolicy, QComboBox
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPalette
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QSize, QObject
import numpy as np
import base64

# Assuming you have edge_tts_util.py in the same directory or in your Python path
from edge_tts_util import speak

# Constants for Server Communication
SERVER_ADDRESS = "http://192.168.1.2:5000"  # Replace with your PC server's IP address

# ============ COLOR & LOGO SETTINGS ==============
MAROON_COLOR = "#8D1B3D"
HEADER_COLOR = "#8D1B3D"
FOOTER_COLOR = "#2F2F2F"
LOGO_PATH = "qatar_logo.png"  # Replace with the actual path on your Pi

# ================ Frame Buffer ================
class FrameBuffer(QObject):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.frame = None

    def update_frame(self, frame):
        self.frame = frame
        self.frame_ready.emit(self.frame)

# ================ CaptureThread =================
class CaptureThread(QThread):
    # Signals for updating the UI
    text_ar_updated = pyqtSignal(str)
    text_en_updated = pyqtSignal(str)
    eyebrow_signal = pyqtSignal(bool)
    mode_changed = pyqtSignal(str)
    status_message = pyqtSignal(str)

    def __init__(self, frame_buffer, parent=None):
        super().__init__(parent)
        self.running = True
        self.server_address = SERVER_ADDRESS
        self.draw_face_mesh = False
        self.frame_buffer = frame_buffer
        self.gesture_control = {
            "nod": True,
            "tilt_left": True,
            "tilt_right": True,
            "eyebrow_raise": True,
            "shake": True
        }
        self.last_update_times = {
            'text_ar': 0,
            'text_en': 0,
            'mode': 0
        }
        self.update_interval = 0.5  # Update every 0.5 seconds
        self.cap = None

    def run(self):
        # Initialize the USB webcam via OpenCV
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            self.status_message.emit("Error: Could not open webcam.")
            return

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.status_message.emit("Error: Failed to capture image.")
                continue

            # Frame is already in BGR; encode it as JPEG with reduced quality
            ret, encoded_frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            if not ret:
                continue

            try:
                now = time.time()
                data_to_send = {}
                if now - self.last_update_times['text_ar'] > self.update_interval:
                    data_to_send['request_text_ar'] = True
                    self.last_update_times['text_ar'] = now

                if now - self.last_update_times['text_en'] > self.update_interval:
                    data_to_send['request_text_en'] = True
                    self.last_update_times['text_en'] = now

                if now - self.last_update_times['mode'] > self.update_interval:
                    data_to_send['request_mode'] = True
                    self.last_update_times['mode'] = now

                response = requests.post(
                    f"{self.server_address}/process_frame",
                    files={"frame": ("frame.jpg", encoded_frame.tobytes(), "image/jpeg")},
                    data=data_to_send,
                    timeout=2
                )
                response.raise_for_status()

                data = response.json()
                if 'final_text_ar' in data:
                    self.text_ar_updated.emit(data.get("final_text_ar", ""))

                if 'final_text_en' in data:
                    self.text_en_updated.emit(data.get("final_text_en", ""))

                if 'mode' in data:
                    self.mode_changed.emit(data.get("mode", "word"))
                self.draw_face_mesh = data.get("draw_face_mesh", False)
                self.gesture_control = data.get("gesture_control", {})

                processed_frame_data = data.get("processed_frame")
                if processed_frame_data:
                    processed_frame_bytes = base64.b64decode(processed_frame_data)
                    processed_frame_np = np.frombuffer(processed_frame_bytes, dtype=np.uint8)
                    processed_frame = cv2.imdecode(processed_frame_np, cv2.IMREAD_COLOR)

                    if processed_frame is not None:
                        self.frame_buffer.update_frame(processed_frame)

            except requests.exceptions.RequestException as e:
                print(f"Error communicating with server: {e}")
                self.status_message.emit(f"Server error: {e}")
                time.sleep(1)

        self.cap.release()

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.wait()

    def toggle_mode(self):
        try:
            response = requests.post(f"{self.server_address}/toggle_mode", timeout=2)
            response.raise_for_status()
            data = response.json()
            self.mode_changed.emit(data.get("mode"))
        except requests.exceptions.RequestException as e:
            print(f"Error toggling mode: {e}")
            self.status_message.emit(f"Server error: {e}")

    def remove_last_char(self):
        try:
            response = requests.post(f"{self.server_address}/remove_last_char", timeout=2)
            response.raise_for_status()
            data = response.json()
            self.text_ar_updated.emit(data.get("text_ar"))
        except requests.exceptions.RequestException as e:
            print(f"Error removing last char: {e}")
            self.status_message.emit(f"Server error: {e}")

    def reset_texts(self):
        try:
            response = requests.post(f"{self.server_address}/reset_texts", timeout=2)
            response.raise_for_status()
            data = response.json()
            self.text_ar_updated.emit(data.get("text_ar", ""))
            self.text_en_updated.emit(data.get("text_en", ""))
        except requests.exceptions.RequestException as e:
            print(f"Error resetting texts: {e}")
            self.status_message.emit(f"Server error: {e}")
    
    def toggle_face_mesh(self):
        try:
            response = requests.post(f"{self.server_address}/toggle_face_mesh", timeout=2)
            response.raise_for_status()
            data = response.json()
            self.draw_face_mesh = data.get("draw_face_mesh", False)
            self.status_message.emit(f"Face Mesh: {'ON' if self.draw_face_mesh else 'OFF'}")
        except requests.exceptions.RequestException as e:
            print(f"Error toggling face mesh: {e}")
            self.status_message.emit(f"Server error: {e}")

    def speak_ar(self):
        try:
            response = requests.get(f"{self.server_address}/get_current_data", timeout=2)
            response.raise_for_status()
            data = response.json()
            text_to_speak = data.get("final_text_ar", "")
            speak(text_to_speak)
        except requests.exceptions.RequestException as e:
            print(f"Error retrieving text for speak_ar: {e}")
            self.status_message.emit(f"Server error: {e}")

    def speak_en(self):
        try:
            response = requests.get(f"{self.server_address}/get_current_data", timeout=2)
            response.raise_for_status()
            data = response.json()
            text_to_speak = data.get("final_text_en", "")
            speak(text_to_speak)
        except requests.exceptions.RequestException as e:
            print(f"Error retrieving text for speak_en: {e}")
            self.status_message.emit(f"Server error: {e}")

    def update_gesture_control(self, gesture, enabled):
        try:
            response = requests.post(
                f"{self.server_address}/update_gesture",
                json={"gesture": gesture, "enabled": enabled},
                timeout=2
            )
            response.raise_for_status()
            self.status_message.emit(f"Gesture {gesture} {'enabled' if enabled else 'disabled'}")
        except requests.exceptions.RequestException as e:
            print(f"Error updating gesture control: {e}")
            self.status_message.emit(f"Server error: {e}")

# ================ DisplayThread =================
class DisplayThread(QThread):
    def __init__(self, frame_buffer, camera_label, parent=None):
        super().__init__(parent)
        self.frame_buffer = frame_buffer
        self.camera_label = camera_label
        self.running = True

    def run(self):
        while self.running:
            if self.frame_buffer.frame is not None:
                # Convert from BGR to RGB for Qt display
                frame = cv2.cvtColor(self.frame_buffer.frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(self.camera_label.width(), self.camera_label.height(), Qt.KeepAspectRatio)
                self.camera_label.setPixmap(scaled_pixmap)
            time.sleep(0.05)  # Approximately 20 FPS

    def stop(self):
        self.running = False
        self.wait()

# ================ MainWindow =================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("General Sign Language Interpreter")
        self.resize(1300, 850)

        # Track current language
        self.current_language = "English"

        # Status bar at the bottom
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(f"background-color: {FOOTER_COLOR}; color: white;")
        self.setStatusBar(self.status_bar)

        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: #FFFFFF;
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

        # Main widget and layout
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

        # Face mesh toggle button
        self.btn_toggle_mesh = QPushButton()
        self.btn_toggle_mesh.setToolTip("Show or hide the face mesh lines on camera feed.")
        self.btn_toggle_mesh.setStyleSheet("background-color: white; color: #8D1B3D;")
        self.btn_toggle_mesh.clicked.connect(self.toggle_face_mesh)
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

        # Create separate widget for gesture controls and add it to the instructions layout
        self.gesture_buttons = {}
        gesture_control_widget = QWidget()
        gesture_control_layout = QVBoxLayout(gesture_control_widget)
        self.gesture_label = QLabel()
        self.gesture_label.setStyleSheet("font-size: 16px; color: #8D1B3D;")
        gesture_control_layout.addWidget(self.gesture_label)
        gestures = ["Nod", "Tilt Left", "Tilt Right", "Eyebrow Raise", "Shake Head"]
        for gesture in gestures:
            btn = QPushButton()
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.clicked.connect(
                lambda checked, g=gesture: self.capture_thread.update_gesture_control(g.lower().replace(" ", "_"), checked)
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
        footer_layout.addStretch()
        main_layout.addWidget(footer_bar)

        # Create a frame buffer and display thread
        self.frame_buffer = FrameBuffer()
        self.display_thread = DisplayThread(self.frame_buffer, self.camera_label)
        self.display_thread.start()

        # ========== Start Capture Thread ==========
        self.capture_thread = CaptureThread(self.frame_buffer)
        self.capture_thread.text_ar_updated.connect(self.update_text_ar)
        self.capture_thread.text_en_updated.connect(self.update_text_en)
        self.capture_thread.eyebrow_signal.connect(self.update_eyebrow_label)
        self.capture_thread.mode_changed.connect(self.update_mode_label)
        self.capture_thread.status_message.connect(self.show_status_message)
        self.capture_thread.start()

        # Initialize UI text for the chosen language (default "English")
        self.change_language()

    def change_language(self):
        """Update the entire UI to the selected language."""
        self.current_language = self.language_combo.currentText()
        tab_widget = self.findChild(QTabWidget)
        if self.current_language == "Arabic":
            self.setWindowTitle(" مترجم لغة الإشارةالعام")
            self.label_app_title.setText(" مترجم لغة الإشارةالعام")
            self.btn_toggle_mesh.setText("إظهار/إخفاء خطوط الشبكية")
            if tab_widget:
                tab_widget.setTabText(0, "تعليمات")
                tab_widget.setTabText(1, "متقدم")
            self.label_instructions.setText(
                "<b>تعليمات الحركات:</b><br>"
                "• إمالة إلى اليسار => إضافة الكلمة المكتشفة (وضع الكلمات)<br>"
                "• إمالة إلى اليمين => نطق النص النهائي<br>"
                "• رفع الحواجب => الترجمة بين العربية والإنجليزية<br>"
                "• إيماءة بالرأس => تبديل وضع الكلمات/الأحرف<br>"
                "• هز بالرأس => إزالة آخر حرف (وضع الأحرف)<br>"
                "• رمشة => غير مكتسب<br><br>"
                "<b>وظائف إضافية:</b><br>"
                "• تكلم عربي / تكلم إنجليزي => قراءة النصوص بصوت عالٍ<br>"
                "• إعادة التعيين => مسح النصوص بالعربية والإنجليزية<br>"
                "• إظهار/إخفاء خطوط الشبكية => إظهار أو إخفاء خطوط Mediapipe<br>"
                "<br><i>استمتع بالتجربة!</i>"
            )
            self.label_advanced.setText(
                "<b>الإعدادات المتقدمة:</b><br>"
                "قريباً..."
            )
            self.gesture_label.setText("<b>التحكم بالحركات:</b>")
            self.gesture_buttons["Nod"].setText("تفعيل إيماءة الرأس")
            self.gesture_buttons["Tilt Left"].setText("تفعيل إمالة اليسار")
            self.gesture_buttons["Tilt Right"].setText("تفعيل إمالة اليمين")
            self.gesture_buttons["Eyebrow Raise"].setText("تفعيل رفع الحواجب")
            self.gesture_buttons["Shake Head"].setText("تفعيل هز الرأس")
            self.label_mode.setText("الوضع الحالي: كلمة")
            self.btn_toggle_mode.setText("تبديل الوضع (كلمة/حرف)")
            self.btn_delete_char.setText("حذف آخر حرف")
            self.btn_speak_ar.setText("تكلم عربي")
            self.btn_speak_en.setText("تكلم إنجليزي")
            self.btn_reset.setText("إعادة التعيين")
            self.label_ar.setText("النص النهائي (عربي):")
            self.label_en.setText("النص النهائي (إنجليزي):")
            self.label_eyebrow.setText("الحواجب: غير مكتشفة")
            self.footer_label.setText("© 2023 - مترجم لغة الإشارة المطور")
            self.status_bar.showMessage("تم التبديل إلى اللغة العربية", 3000)
        else:
            self.setWindowTitle("General Sign Language Interpreter")
            self.label_app_title.setText("General Sign Language Interpreter")
            self.btn_toggle_mesh.setText("Toggle Face Mesh")
            if tab_widget:
                tab_widget.setTabText(0, "Instructions")
                tab_widget.setTabText(1, "Advanced")
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
            self.label_advanced.setText(
                "<b>Advanced Settings:</b><br>"
                "Coming soon..."
            )
            self.gesture_label.setText("<b>Gesture Control:</b>")
            self.gesture_buttons["Nod"].setText("Enable Nod")
            self.gesture_buttons["Tilt Left"].setText("Enable Tilt Left")
            self.gesture_buttons["Tilt Right"].setText("Enable Tilt Right")
            self.gesture_buttons["Eyebrow Raise"].setText("Enable Eyebrow Raise")
            self.gesture_buttons["Shake Head"].setText("Enable Shake Head")
            self.label_mode.setText("Current Mode: Word")
            self.btn_toggle_mode.setText("Toggle Mode (Word/Char)")
            self.btn_delete_char.setText("Delete Last Char")
            self.btn_speak_ar.setText("Speak AR")
            self.btn_speak_en.setText("Speak EN")
            self.btn_reset.setText("Reset All")
            self.label_ar.setText("Final Text (AR):")
            self.label_en.setText("Final Text (EN):")
            self.label_eyebrow.setText("Eyebrows: Not Detected")
            self.footer_label.setText("© 2023 - Enhanced Sign Language Interpreter")
            self.status_bar.showMessage("Switched to English", 3000)

    def update_text_ar(self, text_ar: str):
        if self.current_language == "Arabic":
            self.label_ar.setText("النص النهائي (عربي): " + text_ar)
        else:
            self.label_ar.setText("Final Text (AR): " + text_ar)

    def update_text_en(self, text_en: str):
        if self.current_language == "Arabic":
            self.label_en.setText("النص النهائي (إنجليزي): " + text_en)
        else:
            self.label_en.setText("Final Text (EN): " + text_en)

    def update_eyebrow_label(self, is_raised: bool):
        if self.current_language == "Arabic":
            self.label_eyebrow.setText("الحواجب: مكتشفة" if is_raised else "الحواجب: غير مكتشفة")
        else:
            self.label_eyebrow.setText("Eyebrows: DETECTED" if is_raised else "Eyebrows: Not Detected")

    def update_mode_label(self, mode_str: str):
        if self.current_language == "Arabic":
            if mode_str.lower() == "word":
                self.label_mode.setText("الوضع الحالي: كلمة")
            else:
                self.label_mode.setText("الوضع الحالي: حرف")
        else:
            self.label_mode.setText(f"Current Mode: {mode_str.capitalize()}")

    def show_status_message(self, msg: str):
        self.status_bar.showMessage(msg, 3000)

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

    def toggle_face_mesh(self):
        self.capture_thread.toggle_face_mesh()

    def closeEvent(self, event):
        self.capture_thread.stop()
        self.display_thread.quit()
        self.display_thread.wait()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
