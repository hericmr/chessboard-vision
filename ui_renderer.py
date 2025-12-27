import cv2
import chess
from noise_handler import NoiseState

class UiRenderer:
    """
    Handles all visual feedback and drawing on the frame.
    """
    def __init__(self, grid_extractor, player_color='white'):
        self.grid = grid_extractor
        self.player_color = player_color
        
        # State inputs
        self.fps_display = 0
        self.highlight_squares = [] # [(f, r), ...]
        self.sync_highlight_squares = []
        self.lifted_piece_square = None
        self.radar_destinations = []
        
    def set_player_color(self, color):
        self.player_color = color

    def update_fps(self, fps):
        self.fps_display = fps

    def update_radar(self, game, vision_occupied):
        """Calculates radar (lifted piece dests) based on vision difference"""
        expected_occupied = set()
        if game.board:
             for sq in chess.SQUARES:
                piece = game.board.piece_at(sq)
                if piece:
                     f = chess.square_file(sq)
                     r = chess.square_rank(sq)
                     expected_occupied.add((f, 7-r))
        
        lifted_squares = expected_occupied - vision_occupied
        
        self.lifted_piece_square = None
        self.radar_destinations = []
        
        if len(lifted_squares) == 1:
            pos = list(lifted_squares)[0]
            f, r = pos
            sq_idx = chess.square(f, r)
            piece = game.board.piece_at(sq_idx)
            
            if piece and piece.color == game.board.turn:
                self.lifted_piece_square = pos
                for move in game.board.legal_moves:
                    if move.from_square == sq_idx:
                        dst_f = chess.square_file(move.to_square)
                        dst_r = chess.square_rank(move.to_square)
                        self.radar_destinations.append((dst_f, dst_r))
        
        # Update highligts for drawing
        self.highlight_squares = self.radar_destinations

    def draw(self, vis, board_size, noise_state, img_raw, game_board, sync_highlights):
        sq_size = board_size // 8
        
        # Grid
        if self.grid.grid_lines_x and self.grid.grid_lines_y:
             for x in self.grid.grid_lines_x:
                 cv2.line(vis, (int(x), 0), (int(x), board_size), (0, 200, 100), 1)
             for y in self.grid.grid_lines_y:
                 cv2.line(vis, (0, int(y)), (board_size, int(y)), (0, 200, 100), 1)
        else:
             for i in range(9):
                 cv2.line(vis, (i * sq_size, 0), (i * sq_size, board_size), (50, 50, 50), 1)
                 cv2.line(vis, (0, i * sq_size), (board_size, i * sq_size), (50, 50, 50), 1)
            
        # Noise Overlay
        if noise_state == NoiseState.NOISE_ACTIVE:
            overlay = vis.copy()
            overlay[:] = (0, 0, 80)
            cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
            cv2.putText(vis, "JOGADA EM ANDAMENTO", (board_size//2 - 120, board_size//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # 1. Sync Highlights (Dark Blue - Opponent Move)
        if sync_highlights:
             overlay = vis.copy()
             for (f, r) in sync_highlights:
                 norm_pos = (f, 7-r)
                 raw_pos = self._denormalize_coord(norm_pos) 
                 col, row = raw_pos
                 x1, y1 = col * sq_size, row * sq_size
                 cv2.rectangle(overlay, (x1, y1), (x1+sq_size, y1+sq_size), (100, 50, 0), -1)  # Azul escuro
             cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)

        # 2. Legal Moves Highlights (Dark Green Circles - My Move)
        for dest in self.highlight_squares:
            df, dr = dest
            norm_pos = (df, 7-dr)
            raw_pos = self._denormalize_coord(norm_pos)
            col, row = raw_pos
            
            x1, y1 = col * sq_size, row * sq_size
            cx, cy = x1 + sq_size//2, y1 + sq_size//2
            radius = int(sq_size * 0.4 / 2)  # 40% da casa
            cv2.circle(vis, (cx, cy), radius, (0, 100, 0), -1)  # Verde escuro preenchido

        # Draw Pieces
        if game_board:
            for f in range(8):
                for r in range(8):
                    col, row = f, 7 - r
                    # Adjust for rotation if needed in future, but piece text is always upright
                    # Position calculation needs to respect rotation? 
                    # Actually wait, _denormalize_coord handles the grid slot, but the text position
                    # is based on x, y of the grid slot.
                    
                    # Logic:
                    # Logic Coord: (f, r) -> piece
                    # Visual Slot: norm_pos = (f, 7-r)
                    # Camera Slot: raw_pos = _denormalize(norm_pos)
                    
                    norm_pos = (f, 7-r)
                    raw_pos = self._denormalize_coord(norm_pos)
                    r_col, r_row = raw_pos
                    
                    x = r_col * sq_size + sq_size // 2
                    y = r_row * sq_size + sq_size // 2
                    
                    sq_idx = chess.square(f, r)
                    piece = game_board.piece_at(sq_idx)
                    
                    if piece:
                        sym = piece.symbol()
                        color = (255, 255, 255) if piece.color == chess.WHITE else (0, 0, 0)
                        bg = (0, 0, 0) if piece.color == chess.WHITE else (255, 255, 255)
                        cv2.putText(vis, sym, (x - 15, y + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, bg, 4)
                        cv2.putText(vis, sym, (x - 15, y + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            
            # Status Text
            turn_text = 'Brancas' if game_board.turn else 'Pretas'
            cv2.putText(vis, f"Turno: {turn_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(vis, f"FPS: {self.fps_display:.1f}", (board_size - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Player Perspective Rotation for Display
        if self.player_color == 'black':
            vis = cv2.rotate(vis, cv2.ROTATE_180)
            img_raw = cv2.rotate(img_raw, cv2.ROTATE_180)
                   
        cv2.imshow("Tabuleiro", vis)
        cv2.imshow("Camera", img_raw)

    def _denormalize_coord(self, norm_pos):
        """Convers NORMALIZED (Logic) -> RAW (Camera)"""
        col, row = norm_pos
        if self.player_color == 'black':
            return (7 - col, 7 - row)
        return (col, row)
