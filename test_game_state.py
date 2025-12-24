import unittest
import chess
from game_state import GameState

class TestGameState(unittest.TestCase):
    def setUp(self):
        self.gs = GameState()
        
    def test_initial_occupancy(self):
        occupied = self.gs.get_board_occupancy()
        # Initial board has 32 pieces
        self.assertEqual(len(occupied), 32)
        # Check specific squares
        self.assertIn((0, 0), occupied) # a1 Rook
        self.assertIn((4, 1), occupied) # e2 Pawn
        self.assertNotIn((4, 4), occupied) # e5 Empty
        
    def test_normal_move_logic(self):
        # Move White e2 Pawn to e4
        occupied = self.gs.get_board_occupancy()
        occupied.remove((4, 1)) # e2 becomes empty
        occupied.add((4, 3))    # e4 becomes occupied
        
        move, status = self.gs.process_occupancy_change(occupied)
        
        self.assertEqual(status, "move_confirmed")
        self.assertIsNotNone(move)
        self.assertEqual(move.uci(), "e2e4")
        
        # Verify internal board updated
        self.assertIsNone(self.gs.board.piece_at(chess.E2))
        self.assertIsNotNone(self.gs.board.piece_at(chess.E4))
        
    def test_illegal_move_logic(self):
        # Try e2 to e5 (Pawn moves 3 squares - illegal)
        occupied = self.gs.get_board_occupancy()
        occupied.remove((4, 1)) # e2
        occupied.add((4, 4))    # e5
        
        move, status = self.gs.process_occupancy_change(occupied)
        
        self.assertEqual(status, "illegal_move")
        self.assertIsNone(move)
        
        # Verify board did NOT update
        self.assertIsNotNone(self.gs.board.piece_at(chess.E2))

    def test_turn_switching(self):
        # 1. White e2e4
        occ = self.gs.get_board_occupancy()
        occ.remove((4, 1))
        occ.add((4, 3))
        self.gs.process_occupancy_change(occ)
        self.assertEqual(self.gs.board.turn, chess.BLACK)
        self.assertEqual(self.gs.get_turn_name(), "black")
        
        # 2. Black e7e5
        occ = self.gs.get_board_occupancy()
        occ.remove((4, 6)) # e7
        occ.add((4, 4))    # e5
        move, msg = self.gs.process_occupancy_change(occ)
        self.assertEqual(msg, "move_confirmed")
        self.assertEqual(move.uci(), "e7e5")
        
    def test_capture_logic_simple(self):
        # Setup: e2e4, d7d5, e4xd5
        self.gs.board.push(chess.Move.from_uci("e2e4"))
        self.gs.board.push(chess.Move.from_uci("d7d5"))
        
        # Now White e4 captures d5
        # Visual: e4 vanishes, d5 stays occupied (with white pawn now)
        occ = self.gs.get_board_occupancy()
        occ.remove((4, 3)) # e4 gone
        
        move, status = self.gs.process_occupancy_change(occ)
        
        self.assertEqual(status, "capture_confirmed")
        self.assertEqual(move.uci(), "e4d5")
        
        # Verify d5 occupied by White Pawn
        piece_d5 = self.gs.board.piece_at(chess.D5)
        self.assertIsNotNone(piece_d5)
        self.assertEqual(piece_d5.color, chess.WHITE)

    def test_kingside_castling(self):
        # Setup position where kingside castling is legal
        # Clear squares between king and rook
        self.gs.set_fen("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        
        # Kingside castle: King e1->g1, Rook h1->f1
        # Vanished: (4,0)=e1, (7,0)=h1
        # Appeared: (6,0)=g1, (5,0)=f1
        occ = self.gs.get_board_occupancy()
        occ.remove((4, 0))  # e1 (king gone)
        occ.remove((7, 0))  # h1 (rook gone)
        occ.add((6, 0))     # g1 (king appears)
        occ.add((5, 0))     # f1 (rook appears)
        
        move, status = self.gs.process_occupancy_change(occ)
        
        self.assertEqual(status, "castling_confirmed")
        self.assertEqual(move.uci(), "e1g1")
        
        # Verify king and rook positions
        self.assertEqual(self.gs.board.piece_at(chess.G1).piece_type, chess.KING)
        self.assertEqual(self.gs.board.piece_at(chess.F1).piece_type, chess.ROOK)

    def test_queenside_castling(self):
        # Setup position where queenside castling is legal
        self.gs.set_fen("r3kbnr/pppqpppp/2n5/3p1b2/3P1B2/2N5/PPPQPPPP/R3KBNR w KQkq - 6 5")
        
        # Queenside castle: King e1->c1, Rook a1->d1
        occ = self.gs.get_board_occupancy()
        occ.remove((4, 0))  # e1 (king gone)
        occ.remove((0, 0))  # a1 (rook gone)
        occ.add((2, 0))     # c1 (king appears)
        occ.add((3, 0))     # d1 (rook appears)
        
        move, status = self.gs.process_occupancy_change(occ)
        
        self.assertEqual(status, "castling_confirmed")
        self.assertEqual(move.uci(), "e1c1")

    def test_en_passant(self):
        # Setup: White pawn on e5, Black just played d7d5
        # FEN where en passant is possible
        self.gs.set_fen("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
        
        # En passant: White e5 captures d5 pawn, lands on d6
        # Vanished: (4,4)=e5 (attacker), (3,4)=d5 (victim)
        # Appeared: (3,5)=d6 (attacker destination)
        occ = self.gs.get_board_occupancy()
        occ.remove((4, 4))  # e5 (white pawn gone)
        occ.remove((3, 4))  # d5 (black pawn captured)
        occ.add((3, 5))     # d6 (white pawn appears)
        
        move, status = self.gs.process_occupancy_change(occ)
        
        self.assertEqual(status, "en_passant_confirmed")
        self.assertEqual(move.uci(), "e5d6")
        
        # Verify d5 is empty (captured pawn gone)
        self.assertIsNone(self.gs.board.piece_at(chess.D5))
        # Verify pawn is on d6
        self.assertEqual(self.gs.board.piece_at(chess.D6).piece_type, chess.PAWN)

    def test_get_turn(self):
        self.assertEqual(self.gs.get_turn(), chess.WHITE)
        self.assertEqual(self.gs.get_turn_name(), "white")
        
        self.gs.board.push(chess.Move.from_uci("e2e4"))
        self.assertEqual(self.gs.get_turn(), chess.BLACK)
        self.assertEqual(self.gs.get_turn_name(), "black")

    def test_reset(self):
        self.gs.board.push(chess.Move.from_uci("e2e4"))
        self.gs.reset()
        self.assertEqual(self.gs.get_fen(), chess.STARTING_FEN)

if __name__ == '__main__':
    unittest.main()

