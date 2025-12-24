import chess

class GameState:
    def __init__(self):
        self.board = chess.Board() # Standard starting position
        
    def get_fen(self):
        return self.board.fen()
        
    def get_legal_moves(self):
        return list(self.board.legal_moves)
        
    def get_board_occupancy(self):
        """
        Returns a set of (file, rank) tuples that are currently occupied according to internal logic.
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
        
        vision_occupancy_grid: 8x8 boolean matrix or set of (f,r) occupied squares.
                               Let's assume it's a set of (f,r) tuples for easier comparison.
        """
        
        # 1. Get expected occupancy from Logic
        logical_occupied = self.get_board_occupancy()
        
        # 2. Identify Discrepancies
        # vanished = Logic says Occupied, Vision says Empty (Potential Source)
        vanished = logical_occupied - vision_occupancy_grid
        
        # appeared = Logic says Empty, Vision says Occupied (Potential Dest)
        appeared = vision_occupancy_grid - logical_occupied
        
        # 3. Analyze Patterns
        # Case A: Normal Move (1 vanishes, 1 appears)
        if len(vanished) == 1 and len(appeared) == 1:
            src = list(vanished)[0]
            dst = list(appeared)[0]
            
            move = self._validate_move(src, dst)
            if move:
                self.board.push(move)
                return move, "move_confirmed"
            else:
                return None, "illegal_move"
                
        # Case B: Capture (1 vanished, 0 appeared)
        if len(vanished) == 1 and len(appeared) == 0:
            src = list(vanished)[0]
            src_sq = chess.square(src[0], src[1])
            
            # Find all legal captures from 'src' to an occupied square
            candidates = []
            for move in self.board.legal_moves:
                if move.from_square == src_sq:
                    if self.board.is_capture(move):
                        dst_sq = move.to_square
                        dst_tuple = (chess.square_file(dst_sq), chess.square_rank(dst_sq))
                        
                        # Check if vision *actually* sees it occupied
                        if dst_tuple in vision_occupancy_grid:
                             candidates.append(move)
                             
            if len(candidates) == 1:
                move = candidates[0]
                self.board.push(move)
                return move, "capture_confirmed"
            elif len(candidates) > 1:
                return None, "ambiguous_capture"
        
        return None, "no_valid_change"

    def _validate_move(self, src_tuple, dst_tuple):
        """
        src_tuple: (file, rank) e.g. (0, 1) for a2
        """
        # Convert to sq integer
        src_sq = chess.square(src_tuple[0], src_tuple[1])
        dst_sq = chess.square(dst_tuple[0], dst_tuple[1])
        
        move = chess.Move(src_sq, dst_sq)
        
        if move in self.board.legal_moves:
            return move
            
        # Check promotion (auto-promote to Queen for now?)
        # Vision doesn't see promotion type.
        move_prom = chess.Move(src_sq, dst_sq, promotion=chess.QUEEN)
        if move_prom in self.board.legal_moves:
            return move_prom
            
        return None
