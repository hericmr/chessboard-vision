#!/usr/bin/env python3
"""
Play Lichess - Tabuleiro Físico ↔ Lichess

Integra visão computacional com Lichess API para jogar partidas reais.
"""

import cv2
import numpy as np
import chess
import time
import threading
from queue import Queue

import board_detection
from grid_extractor import SmartGridExtractor
from calibration_module import CalibrationModule
from game_state import GameState
from noise_handler import NoiseHandler, NoiseState
from lichess_client import LichessClient
from piece_detector import PieceDetector

# Configuration
CAMERA_ID = 0
WIDTH, HEIGHT = 1280, 720

class LichessGame:
    """Manages a game between physical board and Lichess."""
    
    def __init__(self):
        self.game = GameState()
        self.lichess = LichessClient()
        self.noise = NoiseHandler()
        self.noise = NoiseHandler()
        self.piece_detector = PieceDetector()  # Detecção de peças circulares
        
        # Threading
        self.move_queue = Queue()
        self.stop_event = threading.Event()
        self.turn_lock = threading.Lock()
        
        # State
        self.my_color = None
        self.game_id = None
        self.waiting_for_opponent = False
        self.last_lichess_moves = ""
        
        # Estabilização temporal
        self.stable_occupancy = None
        self.stable_count = 0
        self.STABILITY_REQUIRED = 20   # Aumentado para 20 frames (~1.5s) para tolerar trocas de peças
        self.last_move_time = 0
        self.MOVE_COOLDOWN = 2.0      # Aumentado para 2.0s para dar tempo da mão sair
        
    def connect_lichess(self) -> bool:
        """Connect to Lichess."""
        return self.lichess.connect()
    
    def select_or_create_game(self) -> str:
        """Let user select or create a game."""
        games = self.lichess.get_ongoing_games()
        
        if games:
            print("\n=== JOGOS EM ANDAMENTO ===")
            for i, g in enumerate(games):
                game_id = g.get("gameId", g.get("id", "?"))
                opponent = g.get("opponent", {}).get("username", "?")
                color = g.get("color", "?")
                is_my_turn = g.get("isMyTurn", False)
                turn_str = " ⬅️ SUA VEZ" if is_my_turn else ""
                print(f"  [{i+1}] {game_id}: vs {opponent} ({color}){turn_str}")
            
            print("\n  [0] Criar novo jogo")
            
            try:
                choice = input("\nEscolha: ").strip()
                if choice == "0":
                    return self._create_new_game()
                else:
                    idx = int(choice) - 1
                    if 0 <= idx < len(games):
                        return games[idx].get("gameId", games[idx].get("id"))
            except:
                pass
        
        return self._create_new_game()
    
    def _create_new_game(self) -> str:
        print("\n=== AGUARDANDO JOGO ===")
        print("Crie um jogo no Lichess (Navegador ou App)")
        print("Procurando... (Ctrl+C para cancelar)")
        
        try:
            while True:
                time.sleep(2)
                games = self.lichess.get_ongoing_games()
                if games:
                    # Auto-select the first game found
                    g = games[0]
                    game_id = g.get("gameId", g.get("id"))
                    opponent = g.get("opponent", {}).get("username", "?")
                    print(f"\n[!] Jogo Encontrado! vs {opponent}")
                    return game_id
                
                print(".", end="", flush=True)
        except KeyboardInterrupt:
            return None
    
    def start_lichess_stream(self, game_id: str):
        """Start streaming game events in background thread."""
        
        def stream_thread():
            for event in self.lichess.stream_game(game_id):
                if self.stop_event.is_set():
                    break
                    
                event_type = event.get("type")
                
                if event_type == "gameFull":
                    # Initial game state
                    self.my_color = self.lichess.my_color
                    state = event.get("state", {})
                    self._sync_moves(state.get("moves", ""))
                    
                elif event_type == "gameState":
                    # Move update
                    moves_str = event.get("moves", "")
                    self._sync_moves(moves_str)
                    
                    status = event.get("status")
                    if status != "started":
                        print(f"\n[Lichess] Jogo terminado: {status}")
                        self.stop_event.set()
        
        thread = threading.Thread(target=stream_thread, daemon=True)
        thread.start()
        return thread
    
    def _sync_moves(self, moves_str: str):
        """Sync local board with Lichess moves."""
        if moves_str == self.last_lichess_moves:
            return
            
        with self.turn_lock:
            # Reset board and replay all moves
            self.game.reset()
            
            if moves_str:
                for uci in moves_str.split():
                    try:
                        move = chess.Move.from_uci(uci)
                        if move in self.game.board.legal_moves:
                            self.game.board.push(move)
                    except:
                        pass
            
            self.last_lichess_moves = moves_str
            
            # Check whose turn
            is_my_turn = self.lichess.is_my_turn(moves_str)
            self.waiting_for_opponent = not is_my_turn
            
            if not is_my_turn:
                last_move = self.lichess.get_last_move(moves_str)
                if last_move:
                    print(f"\n[Oponente] {last_move}")
    
    def send_move_to_lichess(self, move: chess.Move) -> bool:
        """Send a detected move to Lichess."""
        if self.waiting_for_opponent:
            print("[!] Não é sua vez!")
            return False
            
        with self.turn_lock:
            uci = move.uci()
            if self.lichess.make_move(uci):
                self.last_lichess_moves += f" {uci}" if self.last_lichess_moves else uci
                self.waiting_for_opponent = True
                return True
            return False


def main():
    print("=" * 50)
    print("  TABULEIRO FÍSICO ↔ LICHESS")
    print("=" * 50)
    
    # Initialize camera FIRST
    print("\n[1/3] Iniciando câmera...")
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)
    
    if not cap.isOpened():
        print("[!] Erro ao abrir câmera")
        return
    
    # Calibration BEFORE connecting
    print("\n[2/3] === CALIBRAÇÃO DO TABULEIRO ===")
    print("    (Prepare seu tabuleiro físico agora)")
    calib = CalibrationModule()
    config = calib.run(cap)
    
    if config is None:
        print("Calibração cancelada.")
        cap.release()
        return
    
    corners = config["corners"]
    orientation_flipped = config.get("orientation_flipped", False)
    
    board_corners = np.array(corners).reshape((4, 1, 2))
    points_ordered = board_detection.reorder(board_corners)
    
    # Initialize vision
    grid = SmartGridExtractor()
    if "grid_lines_x" in config and config["grid_lines_x"]:
        grid.grid_lines_x = config["grid_lines_x"]
        grid.grid_lines_y = config["grid_lines_y"]
        print("Smart Grid Carregado!")
    else:
        print("Usando Grade Linear (Padrao)")

    game = LichessGame()
    
    # Capture initial reference
    for _ in range(10):
        cap.read()
    
    success, img = cap.read()
    if success:
        warped, _, _ = board_detection.warp_image(img, points_ordered)
        if orientation_flipped:
            warped = cv2.rotate(warped, cv2.ROTATE_180)
        squares = grid.split_board(warped)
        game.piece_detector.update_references(squares)
    
    print("\n✓ Tabuleiro calibrado!")
    print("\n[3/3] Conectando ao Lichess...")
    
    # NOW connect to Lichess
    if not game.connect_lichess():
        print("[!] Falha na conexão com Lichess")
        cap.release()
        return
    
    # Select game - NOW user is ready
    game_id = game.select_or_create_game()
    if not game_id:
        cap.release()
        return
    
    game.game_id = game_id
    print(f"\n[Lichess] Entrando no jogo: {game_id}")
    
    # Start Lichess stream
    stream_thread = game.start_lichess_stream(game_id)
    
    # Wait a moment for initial sync
    time.sleep(0.5)
    
    print(f"\n=== JOGO INICIADO ===")
    print(f"Jogando como: {game.my_color or 'aguardando...'}")
    print("Pressione 'q' para sair\n")
    
    # Main loop
    frame_count = 0
    while not game.stop_event.is_set():
        success, img = cap.read()
        if not success:
            break
        
        # Warp board
        warped, _, board_size = board_detection.warp_image(img, points_ordered)
        if orientation_flipped:
            warped = cv2.rotate(warped, cv2.ROTATE_180)
        
        squares = grid.split_board(warped)
        sq_size = board_size // 8
        
        # Detect changes phase eliminated (integrated into PieceDetector)
        
        # --- SMART SCAN LOGIC ---
        # Priorizar casas relevantes para o jogo
        squares_to_check = None # Default: scan completo
        frame_count += 1
        
        # A cada 30 frames (1s), scan completo de segurança
        if frame_count % 30 != 0 and game.game.board:
             squares_to_check = set()
             
             # 1. Casas ocupadas atualmente (origens potenciais)
             # Precisamos saber se a peça ainda está lá
             ocupadas = game.game.get_board_occupancy()
             squares_to_check.update(ocupadas)
             
             # 2. Casas de destino de movimentos legais
             # (Para detectar chegadas)
             for move in game.game.board.legal_moves:
                 # Converter square index (0-63) para (file, rank)
                 to_sq = move.to_square
                 
                 # python-chess: file 0=a, rank 0=1
                 # Nossa grid: file 0=a, rank 0=8 (topo da imagem)
                 # Precisamos alinhar coordenadas.
                 # O game_state deve lidar com isso, mas aqui precisamos passar (file, rank)
                 # de acordo com o que o PieceDetector espera.
                 
                 f = chess.square_file(to_sq)
                 r = chess.square_rank(to_sq)
                 
                 # Ajuste de rank: chess(0)=rank 1. visual(0)=rank 8 (topo).
                 # Se a imagem não for flipada, rank visual 0 é rank 8 do xadrez.
                 # Se for flipada (pretas), rank visual 0 é rank 1.
                 
                 # Melhor abordagem: Adicionar TODAS as casas se tiver dúvida,
                 # ou confiar no mapeamento do game_state.
                 # Por segurança, vamos mapear corretamente:
                 
                 visual_file = f
                 visual_rank = 7 - r # Padrão (Brancas embaixo -> a1 é (0,7))
                 
                 # Se estiver jogando de pretas (tabuleiro girado), a lógica inverte?
                 # O PieceDetector indexa por posição no grid da imagem.
                 # Com imagem rotacionada 180, a casa (0,0) visual (topo-esq) passa a ser h1 (7,0)?
                 # Não, se rotacionou a imagem, o topo-esquerda visual é a casa h1 (se pretas embaixo).
                 
                 # SIMPLIFICAÇÃO: Usar game_state para converter sq -> (file, rank)
                 # Mas ele usa coordenadas lógicas.
                 
                 # Vamos assumir conversão padrão: 7-r para rank.
                 squares_to_check.add((visual_file, visual_rank))
        
        # Detect detect pieces (com Smart Scan)
        # Detect detect pieces (com Smart Scan) e mudanças visuais
        piece_detections, visual_changes = game.piece_detector.detect_all_pieces(
            squares, 
            use_delta=True, 
            squares_to_check=squares_to_check
        )
        vision_occupied = {pos for pos, info in piece_detections.items() if info['has_piece']}
        
        # Process through noise handler (usando visual_changes do PieceDetector)
        noise_state, noise_data = game.noise.process(visual_changes)
        
        # Detect lifted piece by comparing expected vs visual occupancy
        lifted_piece_square = None
        legal_destinations = []
        
        # Comparar ocupação esperada (lógica) com visual (PieceDetector)
        expected_occupied = game.game.get_board_occupancy()
        
        # Peças que estão no tabuleiro lógico mas NÃO na visão = levantadas
        lifted_squares = expected_occupied - vision_occupied
        
        if len(lifted_squares) == 1 and not game.waiting_for_opponent:
            pos = list(lifted_squares)[0]
            f, r = pos
            sq_idx = chess.square(f, r)
            piece = game.game.board.piece_at(sq_idx)
            
            if piece and piece.color == game.game.board.turn:
                lifted_piece_square = pos
                # Get all legal moves from this square
                for move in game.game.board.legal_moves:
                    if move.from_square == sq_idx:
                        dst_f = chess.square_file(move.to_square)
                        dst_r = chess.square_rank(move.to_square)
                        legal_destinations.append((dst_f, dst_r))
        
        # Draw visualization
        vis = warped.copy()
        
        # Grid lines
        for i in range(9):
            cv2.line(vis, (i * sq_size, 0), (i * sq_size, board_size), (50, 50, 50), 1)
            cv2.line(vis, (0, i * sq_size), (board_size, i * sq_size), (50, 50, 50), 1)
        
        # NÃO desenhar círculos frame-a-frame (instável)
        # A visualização é feita pelas peças do tabuleiro lógico abaixo
        
        # Noise overlay
        if noise_state == NoiseState.NOISE_ACTIVE:
            overlay = vis.copy()
            overlay[:] = (0, 0, 80)
            cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
        
        # Waiting for opponent overlay
        if game.waiting_for_opponent:
            cv2.putText(vis, "AGUARDANDO OPONENTE", (50, board_size//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Draw lifted piece highlight (red with transparency)
        if lifted_piece_square:
            lf, lr = lifted_piece_square
            col, row = lf, 7 - lr
            x1, y1 = col * sq_size, row * sq_size
            x2, y2 = x1 + sq_size, y1 + sq_size
            overlay = vis.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 200), -1)  # Red fill
            cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)
        
        # Draw legal move highlights (purple/blue with transparency)
        for dest in legal_destinations:
            df, dr = dest
            col, row = df, 7 - dr
            x1, y1 = col * sq_size, row * sq_size
            x2, y2 = x1 + sq_size, y1 + sq_size
            overlay = vis.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (180, 100, 150), -1)  # Purple/blue
            cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)
        
        # Draw pieces
        for f in range(8):
            for r in range(8):
                col, row = f, 7 - r
                x = col * sq_size + sq_size // 2
                y = row * sq_size + sq_size // 2
                
                sq_idx = chess.square(f, r)
                piece = game.game.board.piece_at(sq_idx)
                
                if piece:
                    sym = piece.symbol()
                    color = (255, 255, 255) if piece.color == chess.WHITE else (0, 0, 0)
                    bg = (0, 0, 0) if piece.color == chess.WHITE else (255, 255, 255)
                    cv2.putText(vis, sym, (x - 15, y + 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, bg, 4)
                    cv2.putText(vis, sym, (x - 15, y + 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        
        # Process move using PieceDetector com ESTABILIZAÇÃO TEMPORAL
        expected_occupied = game.game.get_board_occupancy()
        
        # Casas que mudaram
        diff_missing = expected_occupied - vision_occupied  # Peças que saíram
        diff_extra = vision_occupied - expected_occupied    # Peças que apareceram
        
        total_diff = len(diff_missing) + len(diff_extra)
        
        # Verificar estabilidade da ocupação
        # Se houver MUITAS mudanças (> 4), é provavel que seja a mão passando -> RESET
        if total_diff > 4:
            game.stable_count = 0
            game.stable_occupancy = set() # Invalidar
        elif game.stable_occupancy == vision_occupied:
            game.stable_count += 1
        else:
            game.stable_occupancy = vision_occupied.copy()
            game.stable_count = 1
        
        # Cooldown entre movimentos
        current_time = time.time()
        cooldown_ok = (current_time - game.last_move_time) > game.MOVE_COOLDOWN
        
        # Movimento detectado (Fuzzy Logic):
        # Tentar extrair UM movimento legal das diferenças visuais, ignorando ruído.
        
        detected_move = None
        
        if game.stable_count >= game.STABILITY_REQUIRED and cooldown_ok and not game.waiting_for_opponent and noise_state != NoiseState.NOISE_ACTIVE:
            diff_missing_list = list(diff_missing)
            diff_extra_list = list(diff_extra)
            possible_moves = []
            
            # 1. Movimentos normais (Origem -> Destino Visual detectado)
            # Tenta casar peças que sumiram com casas novas que apareceram
            for orig in diff_missing_list:
                 orig_idx = chess.square(orig[0], orig[1])
                 for dest in diff_extra_list:
                      dest_idx = chess.square(dest[0], dest[1])
                      
                      move_cand = chess.Move(orig_idx, dest_idx) 
                      if move_cand not in game.game.board.legal_moves:
                           # Tentar promoção automática para Rainha
                           move_cand_promo = chess.Move(orig_idx, dest_idx, promotion=chess.QUEEN)
                           if move_cand_promo in game.game.board.legal_moves:
                               move_cand = move_cand_promo
                      
                      if move_cand in game.game.board.legal_moves:
                           possible_moves.append(move_cand)
            
            # 2. Capturas (Origem -> Destino já Ocupado)
            # Se a peça "sumiu" e não tem destino visual (diff_extra), pode ter ido para uma casa ocupada.
            # Mesmo se tiver diff_extra (ruído), verificamos capturas possíveis.
            for orig in diff_missing_list:
                 orig_idx = chess.square(orig[0], orig[1])
                 for move in game.game.board.legal_moves:
                     if move.from_square == orig_idx and game.game.board.is_capture(move):
                         # O destino de uma captura continua visualmente ocupado.
                         # Devemos verificar se a casa destino está em vision_occupied.
                         d_f = chess.square_file(move.to_square)
                         d_r = chess.square_rank(move.to_square)
                         if (d_f, d_r) in vision_occupied:
                               possible_moves.append(move)

            # Filtro de unicidade: Se houver apenas UM movimento legal plausível, execute-o.
            unique_moves = list(set(possible_moves))
            
            if len(unique_moves) == 1:
                detected_move = unique_moves[0]
            elif len(unique_moves) > 1:
                print(f"[Ambíguo] Multiplos movimentos: {unique_moves}")

        
        if detected_move:
             print(f">>> MOVIMENTO ROBUSTO: {detected_move.uci()}")
             game.game.board.push(detected_move) # Atualiza tabuleiro local
             
             # Send to Lichess
             if game.send_move_to_lichess(detected_move):
                 print(f"    Enviado ao Lichess ✓")
                 game.last_move_time = current_time
             else:
                 print(f"    [!] Falha ao enviar")
                 game.game.board.pop() # Rollback
                     
             # Update reference (força sincronia visual)
             game.piece_detector.update_references(squares)
             game.noise.reset()
             game.stable_count = 0
        
        # Status
        turn = "VOCÊ" if not game.waiting_for_opponent else "OPONENTE"
        color_str = game.my_color or "?"
        cv2.putText(vis, f"Cor: {color_str.upper()} | Turno: {turn}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # Show
        cv2.imshow("Tabuleiro", vis)
        cv2.imshow("Camera", img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            game.piece_detector.update_references(squares)
            print("[RECALIBRADO]")
    
    # Cleanup
    game.stop_event.set()
    cap.release()
    cv2.destroyAllWindows()
    print("\n[Fim]")


if __name__ == "__main__":
    main()
