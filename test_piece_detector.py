#!/usr/bin/env python3
"""
Teste do PieceDetector - Visualiza detecção de peças circulares

Mostra:
- Círculo VERDE em casas com peça detectada
- Quadrado CINZA em casas vazias
- Confiança e método usado
"""

import cv2
import numpy as np

import board_detection
from grid_extractor import GridExtractor
from calibration_module import CalibrationModule
from piece_detector import PieceDetector

CAMERA_ID = 0
WIDTH, HEIGHT = 1280, 720


def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)
    
    # Calibração do tabuleiro
    print("=== CALIBRAÇÃO DO TABULEIRO ===")
    calib = CalibrationModule()
    config = calib.run(cap)
    
    if config is None:
        cap.release()
        return
    
    corners = config["corners"]
    orientation_flipped = config.get("orientation_flipped", False)
    board_corners = np.array(corners).reshape((4, 1, 2))
    points_ordered = board_detection.reorder(board_corners)
    
    grid = GridExtractor()
    detector = PieceDetector()
    
    print("\n=== TESTE DE DETECÇÃO DE PEÇAS ===")
    print("Verde = peça detectada")
    print("Pressione 'q' para sair\n")
    
    while True:
        success, img = cap.read()
        if not success:
            break
        
        warped, _, board_size = board_detection.warp_image(img, points_ordered)
        if orientation_flipped:
            warped = cv2.rotate(warped, cv2.ROTATE_180)
        
        squares = grid.split_board(warped)
        sq_size = board_size // 8
        
        # Detectar peças
        results = detector.detect_all_pieces(squares)
        
        # Visualização
        vis = warped.copy()
        
        # Grid
        for i in range(9):
            cv2.line(vis, (i * sq_size, 0), (i * sq_size, board_size), (50, 50, 50), 1)
            cv2.line(vis, (0, i * sq_size), (board_size, i * sq_size), (50, 50, 50), 1)
        
        # Desenhar detecções
        piece_count = 0
        for (f, r), info in results.items():
            col, row = f, 7 - r
            center_x = col * sq_size + sq_size // 2
            center_y = row * sq_size + sq_size // 2
            
            if info['has_piece']:
                piece_count += 1
                radius = info.get('radius') or (sq_size // 3)
                conf = info['confidence']
                method = info.get('method', '?')
                is_ellipse = info.get('is_ellipse', False)
                
                # Cor baseada no método
                if method == 'ellipse':
                    color = (0, 200, 255)  # Laranja para elipse
                else:
                    color = (0, 255, 0)    # Verde para círculo
                
                # Desenhar elipse ou círculo
                if is_ellipse and info.get('axes'):
                    axes = info['axes']
                    cv2.ellipse(vis, (center_x, center_y), 
                               (int(axes[0]/2), int(axes[1]/2)), 0, 0, 360, color, 2)
                else:
                    cv2.circle(vis, (center_x, center_y), radius, color, 2)
                
                # Confiança e método
                cv2.putText(vis, f"{conf:.0%}", (center_x - 12, center_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            else:
                # Casa vazia - pequeno X
                cv2.drawMarker(vis, (center_x, center_y), (100, 100, 100),
                              cv2.MARKER_CROSS, 10, 1)
        
        # Status
        cv2.putText(vis, f"Pecas detectadas: {piece_count}/32", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Deteccao de Pecas", vis)
        cv2.imshow("Camera", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
