#!/usr/bin/env python3
"""
Calibrador de Sensibilidade - Janela Única

Tabuleiro + Controles + Info em uma única janela integrada.

Controles:
    Trackbars: Ajuste os parâmetros
    'c': Recalibrar referência
    's': Salvar configurações
    'h': Modo teste de mão (toggle)
    'q': Sair
"""

import cv2
import numpy as np
import json
import os
import chess

import board_detection
from grid_extractor import GridExtractor
from calibration_module import CalibrationModule
from change_detector import ChangeDetector
from game_state import GameState

CAMERA_ID = 0
WIDTH, HEIGHT = 1280, 720
SETTINGS_FILE = "sensitivity_settings.json"

DEFAULT_SETTINGS = {
    "sensitivity": 25,
    "blur_kernel": 5,
    "stable_frames": 10,
    "z_threshold": 2.0,
    "alpha": 0.20,
    "initial_variance": 100,
    "use_gaussian": True,
}


def nothing(x):
    pass


def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            data = json.load(f)
            result = DEFAULT_SETTINGS.copy()
            result.update(data)
            return result
    return DEFAULT_SETTINGS.copy()


def save_settings(settings):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)
    print(f"[SAVED] Configurações salvas!")


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
    detector = ChangeDetector()
    game = GameState()  # Para calcular movimentos legais
    
    # Single window with trackbars
    window_name = "Calibrador de Sensibilidade"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 600)
    
    # Trackbars - intuitive naming
    cv2.createTrackbar("Sensibilidade", window_name, 
                       int((3.0 - settings["z_threshold"]) * 20), 50, nothing)
    cv2.createTrackbar("Tolerancia", window_name, 
                       int(settings["initial_variance"] / 10), 80, nothing)
    cv2.createTrackbar("Velocidade", window_name, 
                       int(settings["alpha"] * 100), 50, nothing)
    cv2.createTrackbar("Suavizacao", window_name, 
                       settings["blur_kernel"], 15, nothing)
    
    print("\n" + "="*50)
    print("CALIBRADOR - JANELA UNICA")
    print("="*50)
    print("'s'=Salvar  'c'=Recalibrar  'h'=Teste mao  'q'=Sair")
    print("="*50 + "\n")
    
    frame_count = 0
    hand_test_mode = False
    hand_test_stats = {"total": 0, "with_changes": 0}
    
    while True:
        success, img = cap.read()
        if not success:
            break
        
        # Read trackbars
        sens_inv = cv2.getTrackbarPos("Sensibilidade", window_name)
        z_thresh = 3.0 - (sens_inv / 20.0)
        z_thresh = max(0.5, min(3.0, z_thresh))
        
        var_scaled = cv2.getTrackbarPos("Tolerancia", window_name)
        init_var = max(10, var_scaled * 10)
        
        speed = cv2.getTrackbarPos("Velocidade", window_name)
        alpha = max(0.01, speed / 100.0)
        
        blur = max(1, cv2.getTrackbarPos("Suavizacao", window_name))
        
        # Update settings
        settings["z_threshold"] = z_thresh
        settings["initial_variance"] = init_var
        settings["alpha"] = alpha
        settings["blur_kernel"] = blur
        
        # Update detector
        detector.z_threshold = z_thresh
        detector.initial_variance = init_var
        detector.alpha = alpha
        detector.blur_kernel = blur
        detector._kernel = max(1, blur | 1)
        
        # Warp board
        warped, _, board_size = board_detection.warp_image(img, points_ordered)
        if orientation_flipped:
            warped = cv2.rotate(warped, cv2.ROTATE_180)
        
        squares = grid_extractor.split_board(warped)
        sq_size = board_size // 8
        
        # Auto-calibrate
        if frame_count == 30 and not detector.is_calibrated:
            print("[AUTO] Referência capturada")
            detector.calibrate(squares)
        frame_count += 1
        
        # Detect changes with detailed intensity analysis
        detailed = detector.detect_changes_detailed(squares) if detector.is_calibrated else {}
        changes = detector.detect_changes(squares) if detector.is_calibrated else {}
        n_changes = len(detailed)
        max_z = max([info['z_score'] for info in detailed.values()]) if detailed else 0
        
        # Classify hand pattern
        pattern = detector.classify_hand_pattern(detailed) if detailed else {}
        
        # Hand test stats
        if hand_test_mode and detector.is_calibrated:
            hand_test_stats["total"] += 1
            if n_changes > 0:
                hand_test_stats["with_changes"] += 1
        
        # === BUILD COMPOSITE IMAGE ===
        board_display = warped.copy()
        
        # Check for lifted piece (1 PARCIAL change) and show legal destinations
        legal_destinations = []
        lifted_square = None
        move_candidates = pattern.get('move_candidates', set())
        
        if len(move_candidates) == 1 and not pattern.get('is_hand'):
            lifted_square = list(move_candidates)[0]
            f, r = lifted_square
            sq_idx = chess.square(f, r)
            piece = game.board.piece_at(sq_idx)
            
            if piece and piece.color == game.board.turn:
                for move in game.board.legal_moves:
                    if move.from_square == sq_idx:
                        dest_f = chess.square_file(move.to_square)
                        dest_r = chess.square_rank(move.to_square)
                        legal_destinations.append((dest_f, dest_r))
        
        # Draw legal destinations overlay (blue)
        if legal_destinations:
            overlay = board_display.copy()
            for dest in legal_destinations:
                df, dr = dest
                dc, drow = df, 7 - dr
                cv2.rectangle(overlay,
                    (dc * sq_size, drow * sq_size),
                    ((dc + 1) * sq_size, (drow + 1) * sq_size),
                    (255, 150, 0), -1)  # Blue fill
            cv2.addWeighted(overlay, 0.3, board_display, 0.7, 0, board_display)
        
        # Draw changes with intensity-based colors
        for (f, r), info in detailed.items():
            col, row = f, 7 - r
            x = col * sq_size + sq_size // 2
            y = row * sq_size + sq_size // 2
            
            intensity = info['intensity']
            pct = info['pct_changed']
            is_circular = info.get('is_circular', False)
            center_ratio = info.get('center_ratio', 1.0)
            
            # Color based on intensity AND circularity
            if intensity == 'TOTAL':
                rect_color = (0, 0, 255)    # RED = mão/bloqueio total
                text_color = (255, 255, 255)
            elif intensity == 'PARCIAL':
                if is_circular:
                    rect_color = (0, 255, 0)    # GREEN = peça circular!
                    text_color = (0, 0, 0)
                else:
                    rect_color = (0, 255, 255)  # YELLOW = mão/braço
                    text_color = (0, 0, 0)
            else:  # LEVE
                rect_color = (255, 200, 100)  # LIGHT BLUE = sombra
                text_color = (0, 0, 0)
            
            # Desenhar CÍRCULO para peça, RETÂNGULO para mão
            if is_circular:
                # Círculo verde para peça
                center = (col * sq_size + sq_size // 2, row * sq_size + sq_size // 2)
                radius = sq_size // 2 - 4
                cv2.circle(board_display, center, radius, rect_color, 3)
            else:
                # Retângulo para não-circular
                cv2.rectangle(board_display,
                    (col * sq_size + 2, row * sq_size + 2),
                    ((col + 1) * sq_size - 2, (row + 1) * sq_size - 2),
                    rect_color, 3)
            
            # Show percentage
            cv2.putText(board_display, f"{pct:.0f}%", (x - 15, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
            cv2.putText(board_display, f"{pct:.0f}%", (x - 15, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
        
        # Create info panel (right side)
        panel_w = 350
        panel = np.zeros((board_size, panel_w, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)
        
        y = 30
        
        # Title
        cv2.putText(panel, "PARAMETROS", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += 35
        
        # Calculate effective threshold
        sigma = np.sqrt(init_var)
        eff_thresh = z_thresh * sigma
        
        # Parameters with explanations
        params = [
            (f"Sensibilidade: {sens_inv}/50", (255, 200, 100),
             f"  Maior = detecta mudancas menores"),
            (f"Tolerancia: {init_var}", (100, 255, 100),
             f"  Sigma={sigma:.0f} Threshold={eff_thresh:.0f}"),
            (f"Velocidade: {alpha:.2f}", (255, 100, 255),
             f"  Maior = adapta mais rapido"),
            (f"Suavizacao: {blur}", (100, 200, 255),
             f"  Maior = menos ruido"),
        ]
        
        for title, color, desc in params:
            cv2.putText(panel, title, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += 20
            cv2.putText(panel, desc, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            y += 25
        
        y += 10
        cv2.line(panel, (10, y), (panel_w-10, y), (80, 80, 80), 1)
        y += 20
        
        # Status
        cv2.putText(panel, "STATUS", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        y += 30
        
        # Count by intensity
        total_count = sum(1 for info in detailed.values() if info['intensity'] == 'TOTAL')
        parcial_count = sum(1 for info in detailed.values() if info['intensity'] == 'PARCIAL')
        leve_count = sum(1 for info in detailed.values() if info['intensity'] == 'LEVE')
        
        # Pattern-based status
        if pattern.get('is_hand'):
            status_text = "JOGADA em ANDAMENTO..."
            status_color = (0, 0, 255)
        elif pattern.get('is_move'):
            status_text = "MOVIMENTO VALIDO!"
            status_color = (0, 255, 0)
        elif len(pattern.get('move_candidates', set())) == 1:
            status_text = "Peca levantada"
            status_color = (0, 200, 255)
        elif n_changes == 0:
            status_text = "Estavel"
            status_color = (0, 255, 0)
        else:
            status_text = f"Analisando... ({n_changes} casas)"
            status_color = (128, 128, 128)
        
        cv2.putText(panel, status_text, (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        y += 25
        
        # Show intensity breakdown
        if n_changes > 0:
            intensity_text = f"T:{total_count} P:{parcial_count} L:{leve_count}"
            cv2.putText(panel, intensity_text, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            y += 20
        
        # Show legal moves count if piece is lifted
        if legal_destinations:
            cv2.putText(panel, f"Destinos legais: {len(legal_destinations)}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 150, 0), 1)
            y += 25
        
        cv2.putText(panel, f"Max Z-score: {max_z:.2f}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        y += 30
        
        # Hand test mode
        if hand_test_mode:
            y += 10
            cv2.rectangle(panel, (5, y-5), (panel_w-5, y+50), (0, 0, 120), -1)
            cv2.putText(panel, "TESTE DE MAO", (10, y+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if hand_test_stats["total"] > 0:
                pct = (hand_test_stats["with_changes"] / hand_test_stats["total"]) * 100
                cv2.putText(panel, f"Ruido: {pct:.1f}%", (10, y+42),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 200), 1)
            y += 60
        
        y += 20
        cv2.line(panel, (10, y), (panel_w-10, y), (80, 80, 80), 1)
        y += 20
        
        # Controls help
        cv2.putText(panel, "TECLAS", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y += 25
        
        keys = [
            ("'s'", "Salvar config"),
            ("'c'", "Recalibrar"),
            ("'h'", "Teste de mao"),
            ("'q'", "Sair"),
        ]
        for key, desc in keys:
            cv2.putText(panel, f"{key} = {desc}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
            y += 20
        
        # Combine board and panel
        composite = np.hstack([board_display, panel])
        
        # Show
        cv2.imshow(window_name, composite)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_settings(settings)
        elif key == ord('c'):
            print("[RECALIBRATING]")
            detector.initial_variance = init_var
            detector.calibrate(squares)
        elif key == ord('h'):
            hand_test_mode = not hand_test_mode
            hand_test_stats = {"total": 0, "with_changes": 0}
            print(f"[HAND TEST] {'ATIVADO' if hand_test_mode else 'DESATIVADO'}")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
