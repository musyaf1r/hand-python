import sys
import io
import cv2
import mediapipe as mp
import time
import math

# Force UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize webcam dengan resolusi tinggi
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# FPS counter
prev_time = 0

# Settings
show_landmarks = True
show_connections = True
show_finger_status = True
show_gestures = True
show_hud = True
fullscreen = False

# Window settings
window_name = 'Hand Tracking Premium'
default_width = 1280
default_height = 720

# Create window
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, default_width, default_height)

# Nama setiap jari
finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
finger_names_id = ['Jempol', 'Telunjuk', 'Tengah', 'Manis', 'Kelingking']

print("=" * 60)
print("HAND TRACKING PREMIUM - MediaPipe Edition")
print("=" * 60)
print("Kontrol:")
print("  Q / ESC - Keluar")
print("  F - Toggle Fullscreen")
print("  H - Toggle HUD")
print("  L - Toggle Landmarks")
print("  C - Toggle Connections")
print("  S - Toggle Finger Status")
print("  G - Toggle Gestures")
print("=" * 60)

def count_fingers(hand_landmarks, handedness):
    """Menghitung jari yang terangkat"""
    fingers = []
    landmarks = hand_landmarks.landmark
    
    # Deteksi Jempol (berbeda untuk tangan kiri/kanan)
    if handedness == "Right":
        if landmarks[4].x < landmarks[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:
        if landmarks[4].x > landmarks[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
    
    # Deteksi 4 jari lainnya
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    
    for tip, pip in zip(finger_tips, finger_pips):
        if landmarks[tip].y < landmarks[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)
    
    return fingers

def detect_gesture(fingers, finger_count):
    """Deteksi gesture tangan"""
    # fingers = [Thumb, Index, Middle, Ring, Pinky]
    
    if finger_count == 0:
        return "FIST", "✊"
    elif finger_count == 5:
        return "OPEN HAND", "✋"
    elif fingers == [0, 1, 0, 0, 0]:
        return "POINTING", "☝️"
    elif fingers == [1, 1, 0, 0, 0]:
        return "PEACE", "✌️"
    elif fingers == [0, 1, 1, 0, 0]:
        return "TWO FINGERS", "✌️"
    elif fingers == [1, 0, 0, 0, 1]:
        return "ROCK", "🤘"
    elif fingers == [1, 1, 1, 0, 0]:
        return "THREE", "👌"
    elif fingers == [0, 0, 0, 0, 1]:
        return "PINKY UP", "🤙"
    elif fingers == [1, 0, 0, 0, 0]:
        return "THUMBS UP", "👍"
    else:
        return f"{finger_count} FINGERS", "🖐️"

def calculate_distance(p1, p2):
    """Menghitung jarak antara dua titik"""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def draw_hand_info(frame, hand_landmarks, handedness, hand_id, w, h):
    """Menggambar informasi tangan"""
    # Hitung finger status
    fingers = count_fingers(hand_landmarks, handedness)
    finger_count = sum(fingers)
    
    # Deteksi gesture
    gesture_name, gesture_emoji = detect_gesture(fingers, finger_count)
    
    # Posisi pergelangan tangan
    wrist = hand_landmarks.landmark[0]
    x_pos = int(wrist.x * w)
    y_pos = int(wrist.y * h)
    
    # Panel info background
    panel_width = 280
    panel_height = 240
    panel_x = x_pos - 50
    panel_y = y_pos - 280
    
    # Pastikan panel tidak keluar frame
    if panel_x < 0:
        panel_x = 10
    if panel_x + panel_width > w:
        panel_x = w - panel_width - 10
    if panel_y < 0:
        panel_y = 10
    
    # Draw panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), 
                 (panel_x + panel_width, panel_y + panel_height), 
                 (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
    cv2.rectangle(frame, (panel_x, panel_y), 
                 (panel_x + panel_width, panel_y + panel_height), 
                 (0, 255, 255), 2)
    
    # Info text
    y_offset = panel_y + 30
    
    # Hand label
    hand_color = (0, 255, 0) if handedness == "Right" else (255, 100, 255)
    cv2.putText(frame, f'HAND #{hand_id} - {handedness.upper()}', 
               (panel_x + 8, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)
    
    y_offset += 35
    
    # Finger count
    cv2.putText(frame, f'Fingers Up: {finger_count}', 
               (panel_x + 8, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    y_offset += 35
    
    # Gesture
    if show_gestures:
        cv2.putText(frame, f'Gesture: {gesture_name}', 
                   (panel_x + 8, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        y_offset += 30
    
    # Finger status
    if show_finger_status:
        for i, (name_id, status) in enumerate(zip(finger_names_id, fingers)):
            color = (0, 255, 0) if status == 1 else (0, 0, 255)
            status_text = "UP" if status == 1 else "DOWN"
            cv2.putText(frame, f'{name_id}: {status_text}', 
                       (panel_x + 8, y_offset + (i * 25)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame, gesture_name

# Setup hand tracking
with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2) as hands:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Tidak dapat membaca frame dari webcam")
            break
        
        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hand tracking
        results = hands.process(rgb_frame)
        
        detected_gestures = []
        
        # Draw hand landmarks and detect fingers
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_id, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                
                hand_label = handedness.classification[0].label
                
                # Draw landmarks
                if show_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS if show_connections else None,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                
                # Draw hand info
                frame, gesture = draw_hand_info(frame, hand_landmarks, hand_label, hand_id + 1, w, h)
                detected_gestures.append(gesture)
        
        # === HUD ===
        if show_hud:
            hud_width = 450
            hud_height = 200
            
            cv2.rectangle(frame, (0, 0), (hud_width, hud_height), (0, 0, 0), -1)
            cv2.rectangle(frame, (0, 0), (hud_width, hud_height), (0, 255, 255), 3)
            
            cv2.putText(frame, 'HAND TRACKING PREMIUM', (12, 38), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            
            cv2.putText(frame, f'FPS: {int(fps)}', (12, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
            cv2.putText(frame, f'Hands Detected: {num_hands}', (12, 108), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.putText(frame, f'Resolution: {w}x{h}', (12, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            
            # Feature status
            features = f"L:{show_landmarks} C:{show_connections} S:{show_finger_status} G:{show_gestures}"
            cv2.putText(frame, features, (12, 168), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            # Fullscreen indicator
            if fullscreen:
                cv2.putText(frame, "FULLSCREEN", (w - 200, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Bottom controls bar
        control_height = 70
        cv2.rectangle(frame, (0, h - control_height), (w, h), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, h - control_height), (w, h), (0, 255, 255), 2)
        
        controls1 = "Q/ESC: Quit  |  F: Fullscreen  |  H: HUD  |  L: Landmarks"
        controls2 = "C: Connections  |  S: Finger Status  |  G: Gestures"
        
        cv2.putText(frame, controls1, (12, h - 42), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, controls2, (12, h - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Show frame
        cv2.imshow(window_name, frame)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # Q or ESC
            break
        elif key == ord('f'):
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                print("✓ Fullscreen: ON")
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, default_width, default_height)
                print("✓ Fullscreen: OFF")
        elif key == ord('h'):
            show_hud = not show_hud
            print(f"✓ HUD: {'ON' if show_hud else 'OFF'}")
        elif key == ord('l'):
            show_landmarks = not show_landmarks
            print(f"✓ Landmarks: {'ON' if show_landmarks else 'OFF'}")
        elif key == ord('c'):
            show_connections = not show_connections
            print(f"✓ Connections: {'ON' if show_connections else 'OFF'}")
        elif key == ord('s'):
            show_finger_status = not show_finger_status
            print(f"✓ Finger Status: {'ON' if show_finger_status else 'OFF'}")
        elif key == ord('g'):
            show_gestures = not show_gestures
            print(f"✓ Gestures: {'ON' if show_gestures else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
print("\n✓ Program selesai!")