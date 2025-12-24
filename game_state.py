import chess

class GameState:
    def __init__(self):
        self.board = chess.Board()  # Standard starting position
        
    def get_fen(self):
        return self.board.fen()
    
    def get_turn(self):
        """Returns chess.WHITE or chess.BLACK for current turn."""
        return self.board.turn
    
    def get_turn_name(self):
        """Returns 'white' or 'black' for current turn."""
        return "white" if self.board.turn == chess.WHITE else "black"
        
    def get_legal_moves(self):
        return list(self.board.legal_moves)
    
    def get_legal_moves_from(self, file, rank):
        """Get all legal moves from a specific square."""
        src_sq = chess.square(file, rank)
        return [m for m in self.board.legal_moves if m.from_square == src_sq]
        
    def get_board_occupancy(self):
        """
        Returns a set of (file, rank) tuples that are currently occupied.
        0-indexed: a1=(0,0), h8=(7,7)
        """
        occupied = set()
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                f = chess.square_file(square)
                r = chess.square_rank(square)
                occupied.add((f, r))
        return occupied

    def process_occupancy_change(self, vision_occupancy_grid):
        """
        Compares vision occupancy with internal board state to detect moves.
        
        vision_occupancy_grid: set of (f,r) tuples for occupied squares.
        
        Returns: (move, status_string)
        """
        
        # 1. Get expected occupancy from Logic
        logical_occupied = self.get_board_occupancy()
        
        # 2. Identify Discrepancies
        # vanished = Logic says Occupied, Vision says Empty (Potential Source)
        vanished = logical_occupied - vision_occupancy_grid
        
        # appeared = Logic says Empty, Vision says Occupied (Potential Dest)
        appeared = vision_occupancy_grid - logical_occupied
        
        n_vanished = len(vanished)
        n_appeared = len(appeared)
        
        # 3. Analyze Patterns
        
        # Case A: Normal Move (1 vanishes, 1 appears)
        if n_vanished == 1 and n_appeared == 1:
            src = list(vanished)[0]
            dst = list(appeared)[0]
            
            move = self._validate_move(src, dst)
            if move:
                self.board.push(move)
                return move, "move_confirmed"
            else:
                return None, "illegal_move"
        
        # Case B: Castling (2 vanish, 2 appear - king and rook swap)
        if n_vanished == 2 and n_appeared == 2:
            move = self._detect_castling(vanished, appeared)
            if move:
                self.board.push(move)
                return move, "castling_confirmed"
            # Could be other 2-2 pattern, fall through
        
        # Case C: En Passant (2 vanish, 1 appears)
        # Attacker moves diagonally, captured pawn vanishes from adjacent file
        if n_vanished == 2 and n_appeared == 1:
            move = self._detect_en_passant(vanished, appeared)
            if move:
                self.board.push(move)
                return move, "en_passant_confirmed"
                
        # Case D: Capture (1 vanished, 0 appeared)
        if n_vanished == 1 and n_appeared == 0:
            src = list(vanished)[0]
            move = self._detect_capture(src, vision_occupancy_grid)
            if move:
                self.board.push(move)
                return move, "capture_confirmed"
            elif move is None:
                return None, "ambiguous_capture"
        
        return None, "no_valid_change"
    
    def _detect_castling(self, vanished, appeared):
        """
        Detect castling move from 2-vanished, 2-appeared pattern.
        
        Kingside (O-O):  King e1->g1, Rook h1->f1 (white) or e8->g8, h8->f8 (black)
        Queenside (O-O-O): King e1->c1, Rook a1->d1 (white) or e8->c8, a8->d8 (black)
        """
        vanished_list = list(vanished)
        appeared_list = list(appeared)
        
        # Find which square had the king
        for v in vanished_list:
            v_sq = chess.square(v[0], v[1])
            piece = self.board.piece_at(v_sq)
            if piece and piece.piece_type == chess.KING:
                # This is a king move - find its destination
                for a in appeared_list:
                    a_sq = chess.square(a[0], a[1])
                    # King move is 2 squares horizontally for castling
                    if abs(a[0] - v[0]) == 2 and a[1] == v[1]:
                        move = chess.Move(v_sq, a_sq)
                        if move in self.board.legal_moves:
                            return move
        return None
    
    def _detect_en_passant(self, vanished, appeared):
        """
        Detect en passant: 2 pieces vanish (attacker + victim), 1 appears (attacker at dest).
        
        The attacker's pawn moves diagonally, victim pawn disappears from adjacent square.
        """
        vanished_list = list(vanished)
        dst = list(appeared)[0]
        dst_sq = chess.square(dst[0], dst[1])
        
        # Try each vanished square as the attacker source
        for src in vanished_list:
            src_sq = chess.square(src[0], src[1])
            piece = self.board.piece_at(src_sq)
            
            # Must be a pawn doing a diagonal capture
            if piece and piece.piece_type == chess.PAWN:
                move = chess.Move(src_sq, dst_sq)
                if move in self.board.legal_moves:
                    # Check if it's actually en passant
                    if self.board.is_en_passant(move):
                        return move
        return None
    
    def _detect_capture(self, src, vision_occupancy_grid):
        """
        Detect capture: 1 piece vanishes (attacker), lands on already-occupied square.
        """
        src_sq = chess.square(src[0], src[1])
        
        # Find all legal captures from 'src' to an occupied square
        candidates = []
        for move in self.board.legal_moves:
            if move.from_square == src_sq and self.board.is_capture(move):
                dst_sq = move.to_square
                dst_tuple = (chess.square_file(dst_sq), chess.square_rank(dst_sq))
                
                # Check if vision sees it occupied (attacker now there)
                if dst_tuple in vision_occupancy_grid:
                    candidates.append(move)
                     
        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) > 1:
            return None  # Ambiguous
        return False  # No valid capture found

    def _validate_move(self, src_tuple, dst_tuple):
        """
        Validate a simple move from src to dst.
        Handles automatic queen promotion for pawns.
        """
        src_sq = chess.square(src_tuple[0], src_tuple[1])
        dst_sq = chess.square(dst_tuple[0], dst_tuple[1])
        
        move = chess.Move(src_sq, dst_sq)
        
        if move in self.board.legal_moves:
            return move
            
        # Check promotion (auto-promote to Queen)
        # Vision can't distinguish promotion piece type
        move_prom = chess.Move(src_sq, dst_sq, promotion=chess.QUEEN)
        if move_prom in self.board.legal_moves:
            return move_prom
            
        return None
    
    def reset(self):
        """Reset to initial position."""
        self.board.reset()
    
    def set_fen(self, fen):
        """Set board state from FEN string."""
        self.board.set_fen(fen)

