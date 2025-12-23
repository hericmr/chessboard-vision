
import unittest
from move_detector import detect_move

class TestMoveDetector(unittest.TestCase):
    
    def test_simple_pawn_move(self):
        # Start: e2 pawn exists
        fen_start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        # End: e2 empty, e4 occupied by P
        fen_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        
        move = detect_move(fen_start, fen_e4)
        self.assertEqual(move, "e2e4")

    def test_knight_move(self):
        # Start
        fen_start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        # Nf3 (g1 -> f3)
        fen_nf3 = "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1"
        
        move = detect_move(fen_start, fen_nf3)
        self.assertEqual(move, "g1f3")

    def test_capture_ignored(self):
        # User rule: Destination MUST have been empty.
        # If dest was occupied, it's a capture, which this step should IGNORE (return None).
        
        # Setup: White P on e4, Black p on d5
        fen_before = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        # e4 takes d5 (exd5) -> e4 becomes empty, d5 becomes P (was p)
        # Wait, if P takes p, d5 changes from 'p' to 'P'.
        # Is that "Empty -> Occupied"? No. It is "Occupied -> Occupied".
        # So logic should return None.
        fen_after = "rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2"
        
        move = detect_move(fen_before, fen_after)
        self.assertIsNone(move, "Capture should be ignored (dest was not empty)")

    def test_invalid_too_many_changes(self):
        # Two pawns move at once
        fen_start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        fen_two = "rnbqkbnr/pppppppp/8/8/4P3/5P2/PPPP2PP/RNBQKBNR b KQkq - 0 1" # e2->e4 AND f2->f3
        
        move = detect_move(fen_start, fen_two)
        self.assertIsNone(move)
        
    def test_piece_morph_sanity(self):
        # Piece changes identity (e.g. Vision error: P detected as B after move)
        # e2 (P) -> e4 (B)
        fen_start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        # e4 has 'B'
        fen_morph = "rnbqkbnr/pppppppp/8/8/4B3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        
        move = detect_move(fen_start, fen_morph)
        self.assertIsNone(move, "Should reject if piece identity changes during move")

if __name__ == '__main__':
    unittest.main()
