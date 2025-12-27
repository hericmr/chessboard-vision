
import cv2
import numpy as np
import chess
import time
import threading
import board_detection
from grid_extractor import SmartGridExtractor
from calibration_module import CalibrationModule
from frame_enhancer import ImageEnhancer
from game_state import GameState
from noise_handler import NoiseHandler, NoiseState
from piece_detector import PieceDetector

class GameSession:
    # Estados da Sessão
    STATE_IDLE = "IDLE"
    STATE_WAITING = "WAITING"
    STATE_PROCESSING = "PROCESSING"

    # Configurações de Estabilidade
    STABILITY_REQUIRED = 20   # Frames estáveis necessários
    MOVE_COOLDOWN = 2.0      # Segundos de espera após um movimento

    def __init__(self):
        self.status = self.STATE_IDLE
        self.config = None
        self.board_lock = threading.RLock()
        
        # Componentes Core
        self.grid = None
        self.game = None
        self.noise = None
        self.enhancer = None
        self.piece_detector = None # Novo detector 'Gold'
        
        # Dados de Calibração
        self.corners = None
        self.points_ordered = None
        self.player_color = None
        self.orientation_flipped = False
        
        # Estado de Runtime
        self.fps_start = time.time()
        self.frame_count = 0
        self.fps_display = 0
        
        # Controle de Estabilidade de Movimento
        self.stable_occupancy = None
        self.stable_count = 0
        self.last_move_time = 0
        
        # Radar / UI
        self.current_radar_destinations = []
        self.lifted_piece_square = None

    def on_calibration_requested(self, cap):
        """Executa a fase de calibração inicial"""
        print("=== CALIBRAÇÃO DO TABULEIRO ===")
        # Nota: CalibrationModule pode consumir vários frames
        calib = CalibrationModule()
        config = calib.run(cap)
        
        if config is None:
            return False
            
        self.config = config
        self.corners = config["corners"]
        self.player_color = config["player_color"]
        self.orientation_flipped = config.get("orientation_flipped", False)
        
        board_corners = np.array(self.corners).reshape((4, 1, 2))
        self.points_ordered = board_detection.reorder(board_corners)
        
        # Inicializar Componentes
        self.grid = SmartGridExtractor()
        if "grid_lines_x" in config and config["grid_lines_x"]:
            self.grid.grid_lines_x = config["grid_lines_x"]
            self.grid.grid_lines_y = config["grid_lines_y"]
            print("Smart Grid Carregado!")
        else:
            print("Usando Grade Linear (Padrao)")
            
        self.game = GameState()
        self.noise = NoiseHandler()
        self.enhancer = ImageEnhancer()
        self.piece_detector = PieceDetector()
        
        # Capturar referência inicial
        self.capture_reference(cap)
        return True

    def capture_reference(self, cap):
        """Captura e processa a referência inicial"""
        print("Capturando referência inicial...")
        # Estabilização
        for _ in range(10):
            cap.read()
            
        success, img = cap.read()
        if success:
            warped, _, _ = board_detection.warp_image(img, self.points_ordered)
            if self.orientation_flipped:
                warped = cv2.rotate(warped, cv2.ROTATE_180)
            squares = self.grid.split_board(warped)
            
            # Atualiza referências do Detector de Peças
            self.piece_detector.update_references(squares)
            
            self.status = self.STATE_IDLE
            print("Referência capturada. Jogo pronto.")

    def on_frame(self, img):
        """Processa um único frame (Evento Principal)"""
        # Calcular FPS
        self.frame_count += 1
        elapsed = time.time() - self.fps_start
        if elapsed >= 1.0:
            self.fps_display = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start = time.time()

        # Warp
        warped, _, board_size = board_detection.warp_image(img, self.points_ordered)
        if self.orientation_flipped:
            warped = cv2.rotate(warped, cv2.ROTATE_180)
            
        squares = self.grid.split_board(warped)
        
        # --- SMART SCAN LOGIC ---
        # Priorizar casas relevantes para o jogo:
        # 1. Casas ocupadas atualmente (origens)
        # 2. Casas de destino de movimentos legais
        squares_to_check = None
        
        # A cada 30 frames (1s), scan completo de segurança
        if self.frame_count % 30 != 0 and self.game.board:
             squares_to_check = set()
             
             with self.board_lock:
                 # 1. Casas ocupadas
                 ocupadas = self.game.get_board_occupancy()
                 squares_to_check.update(ocupadas)
                 
                 # 2. Destinos legais
                 if self.game.board:
                     for move in self.game.board.legal_moves:
                         to_sq = move.to_square
                         f = chess.square_file(to_sq)
                         r = chess.square_rank(to_sq)
                         # Conversão visual simplificada: 7-r para o rank visual
                         visual_file = f
                         visual_rank = 7 - r 
                         squares_to_check.add((visual_file, visual_rank))
        
        # Detectar peças e mudanças visuais
        piece_detections, visual_changes = self.piece_detector.detect_all_pieces(
            squares, 
            use_delta=True, 
            squares_to_check=squares_to_check
        )
        vision_occupied = {pos for pos, info in piece_detections.items() if info['has_piece']}
        
        # Noise Handler
        noise_state, noise_data = self.noise.process(visual_changes)
        
        if noise_state == NoiseState.NOISE_ACTIVE:
            self.status = self.STATE_WAITING
        else:
            self.status = self.STATE_PROCESSING # Analisando estabilidade

        # Atualizar Radar (UI)
        self._update_radar_ui(vision_occupied)

        # Lógica de Movimento Estável
        self._process_stable_move(vision_occupied, squares, noise_state)

        # Visualização
        self._draw_interface(warped, board_size, noise_state, img)

    def _process_stable_move(self, vision_occupied, squares, noise_state):
        """Processa a detecção de movimento com estabilização temporal"""
        with self.board_lock:
            expected_occupied = self.game.get_board_occupancy()
            
            diff_missing = expected_occupied - vision_occupied
            diff_extra = vision_occupied - expected_occupied    
            total_diff = len(diff_missing) + len(diff_extra)
            
            # Verificar estabilidade da ocupação
            if total_diff > 4: # Muita mudança = provável mão/ruído
                self.stable_count = 0
                self.stable_occupancy = set()
            elif self.stable_occupancy == vision_occupied:
                self.stable_count += 1
            else:
                self.stable_occupancy = vision_occupied.copy()
                self.stable_count = 1
            
            # Cooldown
            current_time = time.time()
            cooldown_ok = (current_time - self.last_move_time) > self.MOVE_COOLDOWN
            
            if self.stable_count >= self.STABILITY_REQUIRED and cooldown_ok and noise_state != NoiseState.NOISE_ACTIVE:
                # _infer_move deve ser chamado sob lock
                detected_move = self._infer_move(diff_missing, diff_extra, vision_occupied)
                
                if detected_move:
                    print(f">>> MOVIMENTO ROBUSTO: {detected_move.uci()}")
                    
                    # Hook para subclasses (ex: enviar para Lichess)
                    # Nota: Isso pode bloquear por um tempo (rede), mas mantemos o lock 
                    # para evitar que a thread de stream atualize o tabuleiro enquanto decidimos.
                    if self.on_move_detected(detected_move):
                        # Verificação final de legalidade antes de aplicar
                        if detected_move in self.game.board.legal_moves:
                            self.game.board.push(detected_move)
                            self.last_move_time = current_time
                            
                            # Sincronia visual forçada pós-move
                            self.piece_detector.update_references(squares)
                            self.noise.reset()
                            self.stable_count = 0
                        else:
                            print(f"[CRITICAL] Movimento {detected_move} tornou-se ilegal/já jogado antes do push!")

    def _infer_move(self, diff_missing, diff_extra, vision_occupied):
        """Tenta inferir UM movimento legal das diferenças visuais"""
        diff_missing_list = list(diff_missing)
        diff_extra_list = list(diff_extra)
        possible_moves = []
        
        # 1. Movimentos normais (Origem -> Destino Visual)
        for orig in diff_missing_list:
             orig_idx = chess.square(orig[0], orig[1])
             for dest in diff_extra_list:
                  dest_idx = chess.square(dest[0], dest[1])
                  
                  move_cand = chess.Move(orig_idx, dest_idx) 
                  if move_cand not in self.game.board.legal_moves:
                       # Auto-Queen Promotion
                       move_cand_promo = chess.Move(orig_idx, dest_idx, promotion=chess.QUEEN)
                       if move_cand_promo in self.game.board.legal_moves:
                           move_cand = move_cand_promo
                  
                  if move_cand in self.game.board.legal_moves:
                       possible_moves.append(move_cand)
        
        # 2. Capturas (Origem -> Destino já Ocupado)
        for orig in diff_missing_list:
             orig_idx = chess.square(orig[0], orig[1])
             for move in self.game.board.legal_moves:
                 if move.from_square == orig_idx and self.game.board.is_capture(move):
                     d_f = chess.square_file(move.to_square)
                     d_r = chess.square_rank(move.to_square)
                     if (d_f, d_r) in vision_occupied:
                           possible_moves.append(move)

        unique_moves = list(set(possible_moves))
        if len(unique_moves) == 1:
            return unique_moves[0]
        elif len(unique_moves) > 1:
            print(f"[Ambíguo] Multiplos movimentos: {unique_moves}")
            return None
        return None

    def on_move_detected(self, move):
        """Hook para subclasses. Retorna True se o movimento deve ser aceito localmente."""
        return True

    def _update_radar_ui(self, vision_occupied):
        # Lógica de Radar visual (destacar peça levantada)
        expected_occupied = self.game.get_board_occupancy()
        lifted_squares = expected_occupied - vision_occupied
        
        self.lifted_piece_square = None
        self.current_radar_destinations = []
        
        if len(lifted_squares) == 1:
            pos = list(lifted_squares)[0]
            f, r = pos
            sq_idx = chess.square(f, r)
            piece = self.game.board.piece_at(sq_idx)
            
            if piece and piece.color == self.game.board.turn:
                self.lifted_piece_square = pos
                for move in self.game.board.legal_moves:
                    if move.from_square == sq_idx:
                        dst_f = chess.square_file(move.to_square)
                        dst_r = chess.square_rank(move.to_square)
                        self.current_radar_destinations.append((dst_f, dst_r))

    def _draw_interface(self, vis, board_size, noise_state, img_raw):
        sq_size = board_size // 8
        
        # Grid
        if self.grid.grid_lines_x and self.grid.grid_lines_y:
             # Smart Grid
             for x in self.grid.grid_lines_x:
                 cv2.line(vis, (int(x), 0), (int(x), board_size), (0, 200, 100), 1)
             for y in self.grid.grid_lines_y:
                 cv2.line(vis, (0, int(y)), (board_size, int(y)), (0, 200, 100), 1)
        else:
             # Regular Grid
             for i in range(9):
                 cv2.line(vis, (i * sq_size, 0), (i * sq_size, board_size), (50, 50, 50), 1)
                 cv2.line(vis, (0, i * sq_size), (board_size, i * sq_size), (50, 50, 50), 1)
            
        # Draw Overlays
        if noise_state == NoiseState.NOISE_ACTIVE:
            overlay = vis.copy()
            overlay[:] = (0, 0, 80)
            cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
            cv2.putText(vis, "jogada em andamento", (board_size//2 - 120, board_size//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # Highlight Last Move (Blue - Both Players)
        with self.board_lock:
            if self.game.board and len(self.game.board.move_stack) > 0:
                last_move = self.game.board.peek()
                overlay = vis.copy()
                
                # Origin square
                f_src = chess.square_file(last_move.from_square)
                r_src = chess.square_rank(last_move.from_square)
                col_src, row_src = f_src, 7 - r_src
                x1, y1 = col_src * sq_size, row_src * sq_size
                cv2.rectangle(overlay, (x1, y1), (x1+sq_size, y1+sq_size), (100, 50, 0), -1)  # Azul escuro
                
                # Destination square
                f_dst = chess.square_file(last_move.to_square)
                r_dst = chess.square_rank(last_move.to_square)
                col_dst, row_dst = f_dst, 7 - r_dst
                x1, y1 = col_dst * sq_size, row_dst * sq_size
                cv2.rectangle(overlay, (x1, y1), (x1+sq_size, y1+sq_size), (100, 50, 0), -1)  # Azul escuro
                
                cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)

        # Lifted Piece & Radar
        if self.lifted_piece_square:
            lf, lr = self.lifted_piece_square
            col, row = lf, 7 - lr
            x1, y1 = col * sq_size, row * sq_size
            overlay = vis.copy()
            cv2.rectangle(overlay, (x1, y1), (x1+sq_size, y1+sq_size), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)
            
        for dest in self.current_radar_destinations:
            df, dr = dest
            col, row = df, 7 - dr
            x1, y1 = col * sq_size, row * sq_size
            cx, cy = x1 + sq_size//2, y1 + sq_size//2
            radius = int(sq_size * 0.4 / 2)  # 40% da casa
            overlay = vis.copy()
            cv2.circle(overlay, (cx, cy), radius, (0, 100, 0), -1)  # Verde escuro preenchido
            cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)  # 60% verde, 40% original

        # Draw Pieces - Needs Lock because accessing board state which might change by stream thread
        with self.board_lock:
            if self.game.board:
                for f in range(8):
                    for r in range(8):
                        col, row = f, 7 - r
                        x = col * sq_size + sq_size // 2
                        y = row * sq_size + sq_size // 2
                        
                        sq_idx = chess.square(f, r)
                        piece = self.game.board.piece_at(sq_idx)
                        
                        if piece:
                            sym = piece.symbol()
                            color = (255, 255, 255) if piece.color == chess.WHITE else (0, 0, 0)
                            bg = (0, 0, 0) if piece.color == chess.WHITE else (255, 255, 255)
                            cv2.putText(vis, sym, (x - 15, y + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, bg, 4)
                            cv2.putText(vis, sym, (x - 15, y + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            
            # Status
            turn_text = 'Brancas' if self.game.board and self.game.board.turn else 'Pretas'
            cv2.putText(vis, f"Turno: {turn_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(vis, f"FPS: {self.fps_display:.1f}", (board_size - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                   
        cv2.imshow("Tabuleiro", vis)
        cv2.imshow("Camera", img_raw)
