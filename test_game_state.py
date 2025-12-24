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
        # Src: e2 (4, 1) -> Empty
        # Dst: e4 (4, 3) -> Occupied
        
        # Simulate Vision Occupancy (Initial - e2 + e4)
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
        # Try moving a2 Pawn to b3 (Capture specific, but let's say just illegal step)
        # Or Knight jump with Rook.
        # Let's try e2 to e5 (Pawn moves 3 squares).
        
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
        
        # 2. Black e7e5
        occ = self.gs.get_board_occupancy()
        occ.remove((4, 6)) # e7
        occ.add((4, 4))    # e5
        move, msg = self.gs.process_occupancy_change(occ)
        self.assertEqual(msg, "move_confirmed")
        self.assertEqual(move.uci(), "e7e5")
        
    def test_capture_logic_simple(self):
        # Setup board where capture is possible
        # e2e4, d7d5, e4xd5
        self.gs.board.push(chess.Move.from_uci("e2e4"))
        self.gs.board.push(chess.Move.from_uci("d7d5"))
        
        # Now White e4 captures d5
        # Logic state: e4=WhitePawn, d5=BlackPawn
        # Vision Output: e4 becomes Empty. d5 stays Occupied (now WhitePawn, but Vision just sees Occupied)
        
        occ = self.gs.get_board_occupancy()
        # Visual change: e4 vanishes. d5 remains (was occupied, is occupied).
        occ.remove((4, 3)) # e4 gone
        # d5 is already in occ, so no change there.
        
        # Logic says: Vanished={e4}, Appeared={}
        # process_occupancy_change currently handles Vanished=1, Appeared=1.
        # It needs to handle Capture.
        
        move, status = self.gs.process_occupancy_change(occ)
        
        # Now we expect success
        self.assertEqual(status, "capture_confirmed")
        self.assertEqual(move.uci(), "e4d5")
        
        # Verify internal board updated
        self.assertIsNone(self.gs.board.piece_at(chess.E4))
        # d5 occupied by White Pawn?
        piece_d5 = self.gs.board.piece_at(chess.D5)
        self.assertIsNotNone(piece_d5)
        self.assertEqual(piece_d5.color, chess.WHITE)
        self.assertEqual(piece_d5.symbol(), 'P') 

if __name__ == '__main__':
    unittest.main()
