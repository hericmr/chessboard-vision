#!/usr/bin/env python3
"""
Enhanced Sensitivity Calibrator

Parameters:
- Sensitivity: How different a square must be to count as changed
- Blur: Reduce noise (higher = more smooth)
- Stable Frames: How many frames to wait before confirming

Controls:
    Trackbars: Adjust parameters
    'c': Capture new reference
    's': Save settings
    'q': Quit
"""

import cv2
import numpy as np
import json
import os

import board_detection
from grid_extractor import GridExtractor
from calibration_module import CalibrationModule

CAMERA_ID = 0
WIDTH, HEIGHT = 1280, 720
SETTINGS_FILE = "sensitivity_settings.json"

DEFAULT_SETTINGS = {
    "sensitivity": 25,
    "blur_kernel": 5,
    "stable_frames": 15,
}


def nothing(x):
    pass


def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            data = json.load(f)
            # Merge with defaults for missing keys
            for key in DEFAULT_SETTINGS:
                if key not in data:
                    data[key] = DEFAULT_SETTINGS[key]
            return data
    return DEFAULT_SETTINGS.copy()


def save_settings(settings):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)
    print(f"[SAVED] Configurações salvas em {SETTINGS_FILE}")


def calculate_diff(img1, img2, blur_kernel):
    """Calculate mean absolute difference."""
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Ensure odd kernel
    k = max(1, blur_kernel | 1)
    gray1 = cv2.GaussianBlur(gray1, (k, k), 0)
    gray2 = cv2.GaussianBlur(gray2, (k, k), 0)
    
    diff = cv2.absdiff(gray1, gray2)
    return np.mean(diff)


def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)
    
    # Board calibration
    calib_module = CalibrationModule()
    config = calib_module.run(cap)
    
    if config is None:
        cap.release()
        return
    
    corners = config["corners"]
    orientation_flipped = config.get("orientation_flipped", False)
    board_corners = np.array(corners).reshape((4, 1, 2))
    points_ordered = board_detection.reorder(board_corners)
    
    grid_extractor = GridExtractor()
    settings = load_settings()
    reference = {}
    
    # Windows
    cv2.namedWindow("Board")
    cv2.namedWindow("Controles", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controles", 400, 200)
    
    # Trackbars
    cv2.createTrackbar("Sensibilidade", "Controles", settings["sensitivity"], 100, nothing)
    cv2.createTrackbar("Blur", "Controles", settings["blur_kernel"], 15, nothing)
    cv2.createTrackbar("Frames Estab.", "Controles", settings["stable_frames"], 50, nothing)
    
    print("\n" + "="*50)
    print("CALIBRADOR DE SENSIBILIDADE")
    print("="*50)
    print("Ajuste os parâmetros:")
    print("  Sensibilidade: Quão diferente = mudança")
    print("  Blur: Reduz ruído (maior = mais suave)")
    print("  Frames Estab.: Frames para confirmar movimento")
    print("")
    print("Teclas: 'c'=Capturar ref, 's'=Salvar, 'q'=Sair")
    print("="*50 + "\n")
    
    frame_count = 0
    
    while True:
        success, img = cap.read()
        if not success:
            break
        
        # Read trackbars
        settings["sensitivity"] = cv2.getTrackbarPos("Sensibilidade", "Controles")
        settings["blur_kernel"] = cv2.getTrackbarPos("Blur", "Controles")
        settings["stable_frames"] = cv2.getTrackbarPos("Frames Estab.", "Controles")
        
        # Warp
        warped, _, board_size = board_detection.warp_image(img, points_ordered)
        if orientation_flipped:
            warped = cv2.rotate(warped, cv2.ROTATE_180)
        
        squares = grid_extractor.split_board(warped)
        vis = warped.copy()
        sq_size = board_size // 8
        
        # Auto-capture after 30 frames
        if frame_count == 30 and not reference:
            print("[AUTO] Referência capturada")
            for pos, sq in squares.items():
                reference[pos] = sq.copy()
        frame_count += 1
        
        # Count changes
        changed = 0
        
        for (f, r), sq_img in squares.items():
            col, row = f, 7 - r
            x = col * sq_size + sq_size // 2
            y = row * sq_size + sq_size // 2
            
            if (f, r) in reference:
                diff = calculate_diff(sq_img, reference[(f, r)], settings["blur_kernel"])
                is_changed = diff > settings["sensitivity"]
                
                if is_changed:
                    changed += 1
                    cv2.rectangle(vis,
                        (col * sq_size + 3, row * sq_size + 3),
                        ((col + 1) * sq_size - 3, (row + 1) * sq_size - 3),
                        (0, 255, 255), 3)
                    cv2.putText(vis, f"{diff:.0f}", (x - 12, y + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
                    cv2.putText(vis, f"{diff:.0f}", (x - 12, y + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        # Stats
        has_ref = len(reference) > 0
        cv2.putText(vis, f"Sens: {settings['sensitivity']} | Blur: {settings['blur_kernel']} | Frames: {settings['stable_frames']}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis, f"Mudancas: {changed}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        if not has_ref:
            cv2.putText(vis, "Aguardando referencia...", (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        cv2.imshow("Board", vis)
        cv2.imshow("Camera", img)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_settings(settings)
        elif key == ord('c'):
            print("[CAPTURING] Nova referência...")
            for pos, sq in squares.items():
                reference[pos] = sq.copy()
            print("[DONE]")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
