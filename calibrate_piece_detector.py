#!/usr/bin/env python3
"""
Calibrador Visual de Detecção de Peças

Permite ajustar parâmetros do PieceDetector em tempo real:
- Raio mínimo/máximo para Hough Circles
- Thresholds de detecção
- Parâmetros de cavalo

Teclas:
  's' = Salvar parâmetros
  'r' = Reset para padrão
  'q' = Sair
"""

import cv2
import numpy as np
import json
import os

import board_detection
from grid_extractor import GridExtractor
from calibration_module import CalibrationModule
from piece_detector import PieceDetector

CAMERA_ID = 0
WIDTH, HEIGHT = 1280, 720
SETTINGS_FILE = "piece_detector_settings.json"


class DetectorCalibrator:
    def __init__(self):
        self.detector = PieceDetector()
        
        # Parâmetros ajustáveis (em percentual * 100 ou valor direto)
        self.params = {
            'min_radius': 20,      # 20% do tamanho da casa
            'max_radius': 55,      # 55% do tamanho da casa
            'hough_param1': 100,   # Canny high threshold
            'hough_param2': 30,    # Acumulador threshold
            'small_min': 12,       # Tower top min radius %
            'small_max': 25,       # Tower top max radius %
            'knight_aspect_min': 120,  # Aspecto mínimo * 100
            'knight_aspect_max': 250,  # Aspecto máximo * 100
            'center_diff_thresh': 25,  # Threshold centro vs borda
        }
        
        self.load_settings()
        self.apply_params()
        
    def load_settings(self):
        """Carrega parâmetros salvos."""
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, 'r') as f:
                    saved = json.load(f)
                    self.params.update(saved)
                    print(f"[Carregado] {SETTINGS_FILE}")
            except:
                pass
    
    def save_settings(self):
        """Salva parâmetros atuais."""
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(self.params, f, indent=2)
        print(f"[Salvo] {SETTINGS_FILE}")
    
    def apply_params(self):
        """Aplica parâmetros ao detector."""
        self.detector.min_radius_ratio = self.params['min_radius'] / 100
        self.detector.max_radius_ratio = self.params['max_radius'] / 100
        
    def create_trackbars(self, window):
        """Cria trackbars para ajuste."""
        cv2.createTrackbar('MinRadius%', window, self.params['min_radius'], 50, 
                          lambda v: self.update_param('min_radius', v))
        cv2.createTrackbar('MaxRadius%', window, self.params['max_radius'], 70, 
                          lambda v: self.update_param('max_radius', v))
        cv2.createTrackbar('Hough P1', window, self.params['hough_param1'], 200, 
                          lambda v: self.update_param('hough_param1', v))
        cv2.createTrackbar('Hough P2', window, self.params['hough_param2'], 100, 
                          lambda v: self.update_param('hough_param2', v))
        cv2.createTrackbar('SmallMin%', window, self.params['small_min'], 30, 
                          lambda v: self.update_param('small_min', v))
        cv2.createTrackbar('SmallMax%', window, self.params['small_max'], 40, 
                          lambda v: self.update_param('small_max', v))
        cv2.createTrackbar('CenterDiff', window, self.params['center_diff_thresh'], 100, 
                          lambda v: self.update_param('center_diff_thresh', v))
    
    def update_param(self, name, value):
        """Atualiza parâmetro."""
        self.params[name] = max(1, value)  # Evitar zero
        self.apply_params()


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
    calibrator = DetectorCalibrator()
    
    # Criar janela com trackbars
    window = "Calibrador de Deteccao"
    cv2.namedWindow(window)
    calibrator.create_trackbars(window)
    
    print("\n=== CALIBRADOR DE DETECÇÃO ===")
    print("Ajuste os sliders para calibrar")
    print("'s' = Salvar | 'r' = Reset | 'q' = Sair\n")
    
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
        results = calibrator.detector.detect_all_pieces(squares)
        
        # Visualização
        vis = warped.copy()
        
        # Grid
        for i in range(9):
            cv2.line(vis, (i * sq_size, 0), (i * sq_size, board_size), (50, 50, 50), 1)
            cv2.line(vis, (0, i * sq_size), (board_size, i * sq_size), (50, 50, 50), 1)
        
        # Contadores por método
        method_counts = {}
        
        # Desenhar detecções
        piece_count = 0
        for (f, r), info in results.items():
            col, row = f, 7 - r
            center_x = col * sq_size + sq_size // 2
            center_y = row * sq_size + sq_size // 2
            
            if info['has_piece']:
                piece_count += 1
                radius = info.get('radius') or (sq_size // 3)
                method = info.get('method', '?')
                conf = info['confidence']
                
                # Contar método
                method_counts[method] = method_counts.get(method, 0) + 1
                
                # Cor baseada no método
                colors = {
                    'hough': (0, 255, 0),      # Verde
                    'tower_top': (0, 255, 255), # Amarelo
                    'knight': (255, 0, 255),    # Magenta
                    'center_diff': (255, 200, 0), # Ciano
                    'symmetry': (200, 200, 200), # Cinza
                }
                color = colors.get(method, (255, 255, 255))
                
                cv2.circle(vis, (center_x, center_y), radius, color, 2)
                cv2.putText(vis, f"{conf:.0%}", (center_x - 12, center_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            else:
                # Casa vazia - pequeno ponto
                cv2.circle(vis, (center_x, center_y), 3, (80, 80, 80), -1)
        
        # Painel de status
        panel_h = 120
        panel = np.zeros((panel_h, board_size, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)
        
        # Status
        cv2.putText(panel, f"Pecas: {piece_count}/32", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Métodos usados
        y = 50
        for method, count in method_counts.items():
            colors = {
                'hough': (0, 255, 0),
                'tower_top': (0, 255, 255),
                'knight': (255, 0, 255),
                'center_diff': (255, 200, 0),
                'symmetry': (200, 200, 200),
            }
            color = colors.get(method, (255, 255, 255))
            cv2.putText(panel, f"{method}: {count}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += 20
        
        # Parâmetros atuais
        cv2.putText(panel, f"Radius: {calibrator.params['min_radius']}-{calibrator.params['max_radius']}%", 
                   (300, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(panel, f"Hough: P1={calibrator.params['hough_param1']} P2={calibrator.params['hough_param2']}", 
                   (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(panel, f"CenterDiff: {calibrator.params['center_diff_thresh']}", 
                   (300, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Teclas
        cv2.putText(panel, "'s'=Salvar  'r'=Reset  'q'=Sair", (10, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Combinar
        combined = np.vstack([vis, panel])
        
        cv2.imshow(window, combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            calibrator.save_settings()
        elif key == ord('r'):
            calibrator.params = {
                'min_radius': 20,
                'max_radius': 55,
                'hough_param1': 100,
                'hough_param2': 30,
                'small_min': 12,
                'small_max': 25,
                'knight_aspect_min': 120,
                'knight_aspect_max': 250,
                'center_diff_thresh': 25,
            }
            calibrator.apply_params()
            print("[Reset] Parâmetros resetados")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
