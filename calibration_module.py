import cv2
import numpy as np
import json
import os
import board_detection
from grid_extractor import SmartGridExtractor

CALIBRATION_FILE = "calibration.json"

class CalibrationModule:
    def __init__(self):
        self.points = []
        self.config = {}

    def run(self, cap):
        """
        Runs the interactive calibration process.
        Returns the configuration dictionary or None if cancelled.
        """
        print("Iniciando Calibracao Interativa...")
        print("Clique nos 4 cantos do tabuleiro na ordem: SE, SD, ID, IE (sentido horario)")
        print("Pressione 'r' para resetar pontos, 'q' para sair.")

        # Check for existing calibration
        if os.path.exists(CALIBRATION_FILE):
             print(f"\n[?] Configuracao salva encontrada em {CALIBRATION_FILE}")
             print("    Deseja carregar? (s=Sim, n=Nao)")
             while True:
                 key = input("    Escolha: ").strip().lower()
                 if key == 's':
                     try:
                         with open(CALIBRATION_FILE, 'r') as f:
                             saved_config = json.load(f)
                         print("[CAM] Configuracao carregada!")
                         return saved_config
                     except Exception as e:
                         print(f"[!] Erro ao carregar: {e}")
                         break
                 elif key == 'n':
                     break
        
        cv2.namedWindow("Calibracao")
        cv2.setMouseCallback("Calibracao", self._mouse_callback)
        
        while True:
            success, img = cap.read()
            if not success:
                print("Falha na camera.")
                return None
                
            display = img.copy()
            
            # Draw points
            for i, ptr in enumerate(self.points):
                cv2.circle(display, tuple(ptr), 5, (0, 0, 255), -1)
                cv2.putText(display, str(i+1), (ptr[0]+10, ptr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw rectangle if 4 points
            if len(self.points) == 4:
                pts = np.array(self.points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(display, [pts], True, (0, 255, 0), 2)
                cv2.putText(display, "Pressione 'ENTER' para confirmar", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Calibracao", display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                cv2.destroyAllWindows()
                return None
            elif key == ord('r'):
                self.points = []
            elif key == 13: # Enter
                if len(self.points) == 4:
                    break
        
        # Step 2: Validate Warp & Color & Grid
        corners = np.array(self.points)
        return self._configure_details(cap, corners)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append([x, y])

    def _configure_details(self, cap, corners):
        """
        Second phase: Verify warp, select color, and refine grid.
        """
        ordered_corners = board_detection.reorder(corners.reshape((4, 1, 2)))
        orientation_flipped = False
        
        # Initialize Smart Grid Extractor
        grid_extractor = SmartGridExtractor(debug=True)
        refined_grid_x = None
        refined_grid_y = None
        
        print("\n--- FASE 2: VERIFICACAO E GRADE ---")
        print("'w' = Jogador eh BRANCAS (Padrao)")
        print("'b' = Jogador eh PRETAS (Inverter)")
        print("'g' = Refinar Grade (Smart Grid)")
        print("'s' = Salvar e Sair")
        print("'q' = Cancelar")
        
        last_grid_refinement_time = 0
        
        while True:
            success, img = cap.read()
            if not success: break
            
            warped, _, board_size = board_detection.warp_image(img, ordered_corners)
            
            if orientation_flipped:
                warped = cv2.rotate(warped, cv2.ROTATE_180)
            
            # Draw Grid Overlay
            display_warped = warped.copy()
            
            if refined_grid_x is not None and refined_grid_y is not None:
                # Draw IRREGULAR grid
                self._draw_irregular_grid(display_warped, refined_grid_x, refined_grid_y)
                cv2.putText(display_warped, "SMART GRID ATIVO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # Draw STANDARD grid
                board_detection.draw_chess_grid_dynamic(display_warped, board_size, 
                                                      "BLACK" if orientation_flipped else "WHITE")
            
            cv2.imshow("Verificacao", display_warped)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                cv2.destroyAllWindows()
                return None
            elif key == ord('w'):
                orientation_flipped = False
                print("Orientacao: BRANCAS (Padrao)")
            elif key == ord('b'):
                orientation_flipped = True
                print("Orientacao: PRETAS (Invertido)")
            elif key == ord('g'):
                print("Refinando grade (Smart Grid)...")
                refined_grid_x, refined_grid_y = grid_extractor.refine_grid(warped)
            elif key == ord('s'):
                player_color = "black" if orientation_flipped else "white"
                
                config = {
                    "corners": corners.tolist(),
                    "player_color": player_color,
                    "orientation_flipped": orientation_flipped,
                    "grid_lines_x": [int(x) for x in refined_grid_x] if refined_grid_x else None,
                    "grid_lines_y": [int(y) for y in refined_grid_y] if refined_grid_y else None
                }
                self._save_config(config)
                cv2.destroyAllWindows()
                return config

    def _draw_irregular_grid(self, img, grid_x, grid_y):
        h, w = img.shape[:2]
        
        # Draw Vertical Lines (X lines)
        for x in grid_x:
            cv2.line(img, (int(x), 0), (int(x), h), (0, 255, 0), 2)
            
        # Draw Horizontal Lines (Y lines)
        for y in grid_y:
            cv2.line(img, (0, int(y)), (w, int(y)), (0, 255, 0), 2)

    def _save_config(self, config):
        try:
            with open(CALIBRATION_FILE, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"Configuracao salva em {CALIBRATION_FILE}")
        except Exception as e:
            print(f"Erro ao salvar: {e}")
