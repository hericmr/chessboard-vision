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
from grid_extractor import SmartGridExtractor
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
            'knight_aspect_max': 250,  # Aspecto máximo * 100
            'center_diff_thresh': 40,  # Threshold centro vs borda (aumentado para evitar sombras)
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
    
    def save_settings(self, results=None, sq_size=None):
        """Salva parâmetros atuais e gera relatório."""
        # 1. Save JSON
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(self.params, f, indent=2)
        print(f"[Salvo] {SETTINGS_FILE}")
        
        # 2. Export Stats
        if results and sq_size:
            self.export_stats(results, sq_size)

    def export_stats(self, results, sq_size):
        """Gera relatório de estatísticas das peças."""
        filename = "piece_stats.txt"
        area_square = sq_size ** 2
        
        with open(filename, 'w') as f:
            f.write(f"=== ESTATISTICAS DE PECAS ({len(results)} casas analisadas) ===\n")
            f.write(f"Square Size: {sq_size}px\n")
            f.write(f"{'CASA':<6} {'STATUS':<10} {'METODO':<15} {'RAIO':<8} {'AREA%':<8} {'BG%':<8} {'CONF'}\n")
            f.write("-" * 80 + "\n")
            
            count = 0
            for (col, row), info in results.items():
                if info['has_piece']:
                    count += 1
                    # Logica de coordenadas visual
                    # (col, row) sao indices 0-7. 
                    # Se pretas, a visualizacao ja rotacionou, mas aqui temos indices brutos do grid.
                    # Vamos salvar como (File, Rank) visual mesmo.
                    
                    file_char = chr(ord('a') + col)
                    rank_num = 8 - row
                    coord = f"{file_char}{rank_num}" # Note: isso assume orientacao padrao na grid split
                    
                    radius = info.get('radius', 0)
                    method = info.get('method', 'N/A')
                    conf = info.get('confidence', 0.0)
                    
                    area_piece = np.pi * (radius ** 2)
                    pct_area = (area_piece / area_square) * 100
                    pct_bg = 100 - pct_area
                    
                    f.write(f"{coord:<6} {'PECA':<10} {method:<15} {radius:<8} {pct_area:<8.1f} {pct_bg:<8.1f} {conf:.2%}\n")
            
            f.write("-" * 80 + "\n")
            f.write(f"Total de pecas detectadas: {count}\n")
            
        print(f"[Relatorio] Salvo em {filename}")
    
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
    
    points_ordered = board_detection.reorder(board_corners)
    
    # Init Smart Grid
    grid = SmartGridExtractor()
    if "grid_lines_x" in config and config["grid_lines_x"]:
        grid.grid_lines_x = config["grid_lines_x"]
        grid.grid_lines_y = config["grid_lines_y"]
        print("Smart Grid Carregado!")
    else:
        print("Usando Grade Linear (Padrao)")

    calibrator = DetectorCalibrator()
    
    # Criar janela com trackbars
    window = "Calibrador de Deteccao"
    cv2.namedWindow(window)
    calibrator.create_trackbars(window)
    
    print("\n=== CALIBRADOR DE DETECÇÃO ===")
    print("Ajuste os sliders para calibrar")
    print("'s'=Save 'r'=Reset 'h'=Hist 'd'=Delta 'q'=Quit\n")
    
    use_smoothing = True
    use_delta = False
    
    while True:
        success, img = cap.read()
        if not success:
            break
        
        warped, _, board_size = board_detection.warp_image(img, points_ordered)
        if orientation_flipped:
            warped = cv2.rotate(warped, cv2.ROTATE_180)
        
        squares = grid.split_board(warped)
        sq_size = board_size // 8
        
        # Detectar peças (toggle delta e smooth)
        results, _ = calibrator.detector.detect_all_pieces(squares, use_delta=use_delta, use_smoothing=use_smoothing)
        
        # Visualização
        vis = warped.copy()
        
        # Grid
        if grid.grid_lines_x and grid.grid_lines_y:
             # Draw Smart Grid (Greenish)
             for x in grid.grid_lines_x:
                 cv2.line(vis, (int(x), 0), (int(x), board_size), (0, 200, 100), 1)
             for y in grid.grid_lines_y:
                 cv2.line(vis, (0, int(y)), (board_size, int(y)), (0, 200, 100), 1)
        else:
             # Draw Linear Grid (Gray)
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
                conf = info.get('confidence', 0.0)
                
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
                
                # Calculate Area %
                area_piece = np.pi * (radius ** 2)
                area_square = sq_size ** 2
                pct_area = (area_piece / area_square) * 100
                pct_bg = 100 - pct_area
                
                cv2.circle(vis, (center_x, center_y), radius, color, 2)
                
                # Show percentages
                # Piece %
                cv2.putText(vis, f"Area:{pct_area:.0f}%", (center_x - 20, center_y - radius - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                # Bg % (inverso)
                cv2.putText(vis, f"Bg:{pct_bg:.0f}%", (center_x - 20, center_y + radius + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
                           
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
        mode_str = "SMOOTH" if use_smoothing else "RAW"
        delta_str = "DELTA ON" if use_delta else "DELTA OFF"
        cv2.putText(panel, f"Pecas: {piece_count}/32 [{mode_str}] [{delta_str}]", (10, 25),
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
        cv2.putText(panel, "'s'=Save 'r'=Reset 'h'=Hist 'd'=Delta 'q'=Quit", (10, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Combinar
        combined = np.vstack([vis, panel])
        
        cv2.imshow(window, combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            use_delta = not use_delta
            print(f"[Toggle] Delta: {use_delta}")
        elif key == ord('h'):
            use_smoothing = not use_smoothing
            print(f"[Toggle] Smoothing: {use_smoothing}")
        elif key == ord('s'):
            calibrator.save_settings(results, sq_size)
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
                'center_diff_thresh': 40,
            }
            calibrator.apply_params()
            print("[Reset] Parâmetros resetados")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
