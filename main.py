import customtkinter as ctk
import cv2
import threading
import time
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
import os
import sys

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and PyInstaller"""
    try:
        base_path = sys._MEIPASS  # PyInstaller sets this at runtime
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class MaskDetectionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.iconbitmap(resource_path("data/icon.ico"))
        self.title("Face-Mask Real-Time-Detection yolo-v8m")
        self.geometry("1000x600")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        # Video & Model state
        self.cap = None
        self.running = False
        self.detecting = False
        self.mask_count = 0
        self.fps = 0.0
        self.model = YOLO(resource_path("data/best.pt"))
        self.model.info()
        self.class_names = self.model.names

        # UI Layout
        self.canvas = ctk.CTkCanvas(self, bg="black", width=960, height=540)
        self.canvas.pack(pady=20)

        button_frame = ctk.CTkFrame(self)
        button_frame.pack(pady=10)

        self.start_feed_btn = ctk.CTkButton(button_frame, text="Start Camera Feed", command=self.start_camera)
        self.start_feed_btn.pack(side="left", padx=5)

        self.stop_feed_btn = ctk.CTkButton(button_frame, text="Stop Camera Feed", command=self.stop_camera_feed)
        self.stop_feed_btn.pack(side="left", padx=5)

        self.start_detect_btn = ctk.CTkButton(button_frame, text="Start Detection", command=self.start_detection)
        self.start_detect_btn.pack(side="left", padx=5)

        self.stop_all_btn = ctk.CTkButton(button_frame, text="Stop All", command=self.stop_all)
        self.stop_all_btn.pack(side="left", padx=5)

        self.status_label = ctk.CTkLabel(self, text="Status: Idle", font=("Arial", 14))
        self.status_label.pack(pady=5)

        self.counter_label = ctk.CTkLabel(self, text="Masks Detected: 0", font=("Arial", 14, "bold"))
        self.counter_label.pack(pady=5)

    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.status_label.configure(text="Status: Camera not found!")
                return
        if not self.running:
            self.running = True
            threading.Thread(target=self.video_loop, daemon=True).start()
            self.status_label.configure(text="Status: Camera started")

    def stop_camera_feed(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.canvas.delete("all")
        self.status_label.configure(text="Status: Camera feed stopped")

    def start_detection(self):
        if not self.cap or not self.running:
            self.status_label.configure(text="Status: Start camera first!")
            return
        self.detecting = True
        self.status_label.configure(text="Status: Detection running...")

    def stop_all(self):
        self.running = False
        self.detecting = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.status_label.configure(text="Status: Stopped")
        self.canvas.delete("all")
        self.counter_label.configure(text="Masks Detected: 0")

    def video_loop(self):
        fps_time = time.time()
        frame_count = 0

        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1
            orig_h, orig_w = frame.shape[:2]
            display_frame = frame.copy()

            self.mask_count = 0

            if self.detecting:
                results = self.model.predict(frame, task='detect', verbose=False)
                for r in results:
                    boxes = r.boxes
                    names = self.class_names

                    for box in boxes:
                        cls = int(box.cls[0])
                        if "mask" in names[cls].lower():
                            self.mask_count += 1
                    display_frame = r.plot()

            now = time.time()
            elapsed = now - fps_time
            if elapsed >= 1:
                self.fps = frame_count / elapsed
                frame_count = 0
                fps_time = now

            # Overlays
            cv2.putText(display_frame, f"FPS: {self.fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Masks: {self.mask_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # Aspect-ratio correct padding
            disp_w, disp_h = 960, 540
            scale = min(disp_w / orig_w, disp_h / orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            resized = cv2.resize(display_frame, (new_w, new_h))

            padded = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
            x_offset = (disp_w - new_w) // 2
            y_offset = (disp_h - new_h) // 2
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

            img = Image.fromarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
            self.canvas.image = imgtk

            self.counter_label.configure(text=f"Masks Detected: {self.mask_count}")

        self.stop_all()

if __name__ == "__main__":
    app = MaskDetectionApp()
    app.mainloop()