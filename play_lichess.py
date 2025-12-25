#!/usr/bin/env python3
"""
Play Lichess - Event Driven Driver
"""

import cv2
from lichess_session import LichessSession

# Configuration
CAMERA_ID = 0
WIDTH, HEIGHT = 1280, 720
SKIP_FRAMES = 2

def main():
    # 1. Initialize Camera
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)
    
    if not cap.isOpened():
        print("Erro ao abrir câmera")
        return

    # 2. Start Session
    session = LichessSession()
    
    # 3. Calibration Phase
    if not session.on_calibration_requested(cap):
        print("Calibração cancelada")
        cap.release()
        return

    # 4. Connect to Lichess
    if not session.connect_and_setup():
        cap.release()
        return
        
    print(f"Jogando como: {session.my_color or 'aguardando scan...'}")
    print("Pressione 'q' para sair\n")
    
    frame_count = 0
    
    # 5. Main Loop
    while True:
        success, img = cap.read()
        if not success:
            break
            
        frame_count += 1
        
        # Optimization: Skip frames
        if SKIP_FRAMES > 1 and frame_count % SKIP_FRAMES != 0:
            cv2.imshow("Camera", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Event: Frame Processing
        session.on_frame(img)
        
        # Input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print("[RECALIBRADO]")
            session.capture_reference(cap)
            
    # Cleanup
    session.on_exit()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
