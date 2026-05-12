import threading
import time
import cv2
import mediapipe as mp
from config import *

class HandTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.tip_ndc = None
        self.target_grid = None
        self.confirmed = False
        self.running = True
        
        self._hold_start = None
        self._hold_prev_pos = None
        
        self.cap = cv2.VideoCapture(0)
        self.model_path = "hand_landmarker.task"
        self.options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=1,
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(self.options)
        
        self.thread = threading.Thread(target=self._tracking_thread, daemon=True)
        self.thread.start()
    
    def is_pointer(self, landmarks):
        """Return True if index is up and middle/ring/pinky are curled."""
        return (landmarks[8].y < landmarks[6].y and
                landmarks[12].y > landmarks[10].y and
                landmarks[16].y > landmarks[14].y and
                landmarks[20].y > landmarks[18].y)
    
    def _tracking_thread(self):
        """Background thread for hand tracking."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect(mp_img)

            if result.hand_landmarks and self.is_pointer(result.hand_landmarks[0]):
                lm = result.hand_landmarks[0]
                tip_x = lm[8].x
                tip_y = lm[8].y

                ndc_x = tip_x * 2.0 - 1.0
                ndc_y = -(tip_y * 2.0 - 1.0)

                grid_col = (1.0 - tip_x) * GRID_SIZE
                grid_row = (1.0 - tip_y) * GRID_SIZE
                target_grid = [grid_row, grid_col]

                with self.lock:
                    self.tip_ndc = (ndc_x, ndc_y)
                    self.target_grid = target_grid
                    self.confirmed = True
            else:
                self._hold_prev_pos = None
                self._hold_start = None
                with self.lock:
                    self.tip_ndc = None
                    self.target_grid = None
                    self.confirmed = False

        self.cap.release()
        self.landmarker.close()
    
    def get_hand_state(self):
        """Thread-safe access to hand state."""
        with self.lock:
            return self.tip_ndc, self.target_grid, self.confirmed
    
    def stop(self):
        """Stop the hand tracking thread."""
        self.running = False
        self.thread.join()
