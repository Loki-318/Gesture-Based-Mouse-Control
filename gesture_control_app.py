import cv2
import mediapipe as mp
import pyautogui
import time
import sys
import threading
from pathlib import Path
import os
import numpy as np

# For system tray functionality
try:
    import pystray
    from PIL import Image, ImageDraw
    HAS_TRAY = True
except ImportError:
    print("pystray and/or PIL not found. Installing system tray dependencies...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pystray", "pillow"])
    import pystray
    from PIL import Image, ImageDraw
    HAS_TRAY = True

class GestureControlApp:
    def __init__(self):
        self.running = False
        self.paused = False
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.5
        )
        
        # Constants
        self.SCROLL_SPEED = 50
        self.CLICK_COOLDOWN = 0.5
        self.last_click_time = time.time()
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Status messages
        self.status_message = "Gesture Control Ready"
        self.mode_message = "No hands detected"
        
        # For text selection
        self.selection_active = False
        self.selection_start = None
        
        # If running as a bundled app, disable PyAutoGUI failsafe
        if getattr(sys, 'frozen', False):
            pyautogui.FAILSAFE = False

    def create_tray_icon(self):
        # Create an icon for the system tray
        icon_image = self.create_icon_image()
        
        # Create a menu
        menu = (
            pystray.MenuItem('Show Window', self.show_window),
            pystray.MenuItem('Pause/Resume', self.toggle_pause),
            pystray.MenuItem('Exit', self.exit_app)
        )
        
        # Create the tray icon
        self.icon = pystray.Icon("gesture_control", icon_image, "Gesture Control", menu)
        self.icon.run_detached()
    
    def create_icon_image(self):
        # Create a simple icon (a colored square)
        width = 64
        height = 64
        color = (0, 120, 212)  # Blue color
        
        image = Image.new('RGB', (width, height), color=color)
        dc = ImageDraw.Draw(image)
        
        # Draw a hand shape or just use a simple shape
        dc.rectangle((15, 15, 49, 49), fill=(255, 255, 255))
        
        return image
    
    def show_window(self, icon, item):
        # This will be called when "Show Window" is clicked
        # We will just make sure the OpenCV window is focused
        pass
    
    def toggle_pause(self, icon, item):
        # Toggle pause state
        self.paused = not self.paused
        if self.paused:
            self.status_message = "PAUSED - Press P to resume"
        else:
            self.status_message = "Gesture Control Active"
    
    def exit_app(self, icon, item):
        # Exit the application
        self.running = False
        if hasattr(icon, 'stop'):
            icon.stop()
    
    def start(self):
        # Initialize video capture
        self.cam = cv2.VideoCapture(0)
        
        # Start the system tray if available
        if HAS_TRAY:
            threading.Thread(target=self.create_tray_icon, daemon=True).start()
        
        self.running = True
        try:
            self.main_loop()
        finally:
            self.cleanup()
    
    def main_loop(self):
        while self.running and self.cam.isOpened():
            # Process key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                self.toggle_pause(None, None)
            
            # Skip processing if paused
            if self.paused:
                # Just show a paused message
                black_screen = 255 * np.ones((400, 600, 3), dtype=np.uint8)
                cv2.putText(black_screen, "PAUSED", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                cv2.putText(black_screen, "Press 'P' to resume", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Gesture Control', black_screen)
                time.sleep(0.1)  # Reduce CPU usage while paused
                continue
            
            # Capture frame
            success, image = self.cam.read()
            if not success:
                print("Failed to capture image")
                time.sleep(0.5)  # Wait a bit before trying again
                continue

            # Process the frame
            self.process_frame(image)
    
    def process_frame(self, image):
        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)
        
        frame_height, frame_width = image.shape[:2]
        
        # Draw guide lines
        cv2.line(image, (0, frame_height//2 - 20), (frame_width, frame_height//2 - 20), (0, 255, 0), 2)
        cv2.line(image, (frame_width//2, frame_height), (frame_width//2, 0), (0, 255, 0), 2)
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.hands.process(image_rgb)

        # Reset mode message
        self.mode_message = "No hands detected"
        
        # Process hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmarks
                self.extract_landmarks(hand_landmarks)
                
                # Process single hand gestures
                if len(results.multi_hand_landmarks) == 1:
                    self.mode_message = "One hand"
                    self.process_single_hand(hand_landmarks, image)
                # Process two hand gestures
                elif len(results.multi_hand_landmarks) == 2:
                    self.mode_message = "Two hands"
                    self.process_two_hands(results.multi_hand_landmarks, image)
                
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        # Add status messages
        cv2.putText(image, self.status_message, (10, frame_height - 20), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, self.mode_message, (frame_width - 200, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Show help text
        cv2.putText(image, "Press 'P' to pause, 'Q' to quit", (10, frame_height - 50), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display the resulting frame
        cv2.imshow('Gesture Control', image)
    
    def extract_landmarks(self, hand_landmarks):
        # Extract finger landmarks
        self.thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        self.thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
        self.index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        self.index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        self.middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        self.middle_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        self.ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        self.ring_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP]
        self.little_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        self.little_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP]
        
        # Check finger states
        self.thumb_extended = self.thumb_tip.y < self.thumb_ip.y
        self.index_extended = self.index_tip.y < self.index_pip.y
        self.middle_extended = self.middle_tip.y < self.middle_pip.y
        self.ring_extended = self.ring_tip.y < self.ring_pip.y
        self.little_extended = self.little_tip.y < self.little_pip.y
        
        # Calculate distance for pinch detection
        self.thumb_tip_x = self.thumb_tip.x
        self.thumb_tip_y = self.thumb_tip.y
        self.middle_tip_x = self.middle_tip.x
        self.middle_tip_y = self.middle_tip.y
        self.distance = ((self.thumb_tip_x - self.middle_tip_x)**2 + (self.thumb_tip_y - self.middle_tip_y)**2)**0.5
    
    def process_single_hand(self, hand_landmarks, image):
        # Mouse control with index finger
        if self.index_extended and not (self.middle_extended or self.ring_extended or self.little_extended):
            cursor_x = int(self.index_tip.x * self.screen_width)
            cursor_y = int(self.index_tip.y * self.screen_height + 30)
            pyautogui.moveTo(cursor_x, cursor_y, duration=0.1, tween=pyautogui.easeOutQuad)
            self.status_message = "Mouse Control"
        
        # Vertical scrolling with index and middle fingers
        elif self.index_extended and self.middle_extended and not (self.ring_extended or self.little_extended):
            hand_y = (self.index_tip.y + self.middle_tip.y) / 2
            
            if hand_y > 0.5:
                pyautogui.scroll(-self.SCROLL_SPEED)
                self.status_message = "Scrolling Down"
            else: 
                pyautogui.scroll(self.SCROLL_SPEED) 
                self.status_message = "Scrolling Up"
        
        # Horizontal scrolling with index and little fingers
        elif self.index_extended and self.little_extended and not (self.middle_extended or self.ring_extended):
            hand_x = (self.index_tip.x + self.little_tip.x) / 2
            
            if hand_x > 0.5:
                pyautogui.hscroll(self.SCROLL_SPEED)
                self.status_message = "Scrolling Right"
            else:
                pyautogui.hscroll(-self.SCROLL_SPEED)
                self.status_message = "Scrolling Left"
        
        # Left click with open hand
        elif all([self.thumb_extended, self.index_extended, self.middle_extended, self.ring_extended, self.little_extended]):
            current_time = time.time()
            if current_time - self.last_click_time >= self.CLICK_COOLDOWN:
                pyautogui.click()
                self.last_click_time = current_time
                self.status_message = "Left Click"
        
        # Right click with pinch gesture
        elif self.distance < 0.05:  # Thumb and middle finger pinch
            current_time = time.time()
            if current_time - self.last_click_time >= self.CLICK_COOLDOWN:
                pyautogui.click(button="right")
                self.last_click_time = current_time
                self.status_message = "Right Click"
    
    def process_two_hands(self, multi_hand_landmarks, image):
        hand1 = multi_hand_landmarks[0]
        hand2 = multi_hand_landmarks[1]
        
        # Check if first hand is open (all fingers extended)
        hand1_all_extended = all([
            hand1.landmark[self.mp_hands.HandLandmark.THUMB_TIP].y < hand1.landmark[self.mp_hands.HandLandmark.THUMB_IP].y,
            hand1.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand1.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
            hand1.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand1.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
            hand1.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].y < hand1.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP].y,
            hand1.landmark[self.mp_hands.HandLandmark.PINKY_TIP].y < hand1.landmark[self.mp_hands.HandLandmark.PINKY_PIP].y
        ])
        
        # Check if second hand has only index extended
        hand2_index_only = (
            hand2.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand2.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y
            and hand2.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > hand2.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
            and hand2.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].y > hand2.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP].y
            and hand2.landmark[self.mp_hands.HandLandmark.PINKY_TIP].y > hand2.landmark[self.mp_hands.HandLandmark.PINKY_PIP].y
        )

        # Check if second hand has index and middle fingers extended
        hand2_index_middle = (
                        hand2.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand2.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y
                        and hand2.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand2.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
                        and hand2.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].y > hand2.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP].y
                        and hand2.landmark[self.mp_hands.HandLandmark.PINKY_TIP].y > hand2.landmark[self.mp_hands.HandLandmark.PINKY_PIP].y
                    )
        
        # Text selection - Open palm and index finger pointing
        if hand1_all_extended and hand2_index_only:
            # If selection hasn't started yet, start it
            if not self.selection_active:
                # Start position is the index finger of the hand with all fingers extended
                start_x = int(hand1.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * self.screen_width)
                start_y = int(hand1.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * self.screen_height)
                
                # Move to start position and press mouse button
                pyautogui.moveTo(start_x, start_y)
                pyautogui.mouseDown()
                
                self.selection_active = True
                self.status_message = "Text Selection Started"
            
            # Continue selection by moving to the current position of the pointing finger
            end_x = int(hand2.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * self.screen_width)
            end_y = int(hand2.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * self.screen_height)
            
            pyautogui.moveTo(end_x, end_y)
            self.status_message = "Text Selection Active"
        
        # Finish selection if it was active but the gesture changed
        elif self.selection_active:
            pyautogui.mouseUp()
            self.selection_active = False
            self.status_message = "Text Selection Completed"
        
        # Zoom functionality - open hand and peace sign
        elif hand1_all_extended and hand2_index_middle:
            fingers_y = (hand2.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y + 
                        hand2.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y) / 2
    
            ZOOM_FACTOR = 1  # Adjust as needed
            
            if fingers_y < 0.5:  # Upper half of screen
                # Zoom in - Ctrl and + key
                pyautogui.keyDown('ctrl')
                pyautogui.press('+')
                pyautogui.keyUp('ctrl')
                self.status_message = "Zoom In"
            else:  # Lower half
                # Zoom out - Ctrl and - key
                pyautogui.keyDown('ctrl')
                pyautogui.press('-')
                pyautogui.keyUp('ctrl')
                self.status_message = "Zoom Out"
    
    def cleanup(self):
        # Release resources
        if hasattr(self, 'cam'):
            self.cam.release()
        if hasattr(self, 'hands'):
            self.hands.close()
        cv2.destroyAllWindows()
        
        # If the selection was active, make sure to release the mouse button
        if self.selection_active:
            pyautogui.mouseUp()
            self.selection_active = False

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

def main():
    app = GestureControlApp()
    app.start()

if __name__ == "__main__":
    main()