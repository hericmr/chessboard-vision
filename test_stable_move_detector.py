import unittest
from stable_move_detector import StableMoveDetector

class TestStableMoveDetector(unittest.TestCase):
    
    def test_startup_stabilization(self):
        detector = StableMoveDetector(stability_threshold=3)
        fen_start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        # Feed 2 frames (not enough)
        self.assertIsNone(detector.process(fen_start))
        self.assertIsNone(detector.process(fen_start))
        
        # 3rd frame -> Confirmed (first state) -> Returns None (no move, just startup)
        self.assertIsNone(detector.process(fen_start))
        
        self.assertEqual(detector.confirmed_fen, fen_start)

    def test_move_detection_flow(self):
        detector = StableMoveDetector(stability_threshold=3)
        fen_start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        # 1. Establish baseline
        for _ in range(3):
            detector.process(fen_start)
            
        # 2. Instability (Simulate hand moving piece)
        fen_noise = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" # Partial
        detector.process(fen_noise)
        detector.process(None) # Lost tracking
        detector.process(fen_noise)
        
        self.assertEqual(detector.confirmed_fen, fen_start, "Confirmed FEN should not change during noise")
        
        # 3. New Stable State (e2e4)
        fen_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        
        # Frame 1
        self.assertIsNone(detector.process(fen_e4))
        # Frame 2
        self.assertIsNone(detector.process(fen_e4))
        # Frame 3 -> CONFIRM
        move = detector.process(fen_e4)
        
        self.assertEqual(move, "e2e4", "Should detect move after stabilization")
        self.assertEqual(detector.confirmed_fen, fen_e4)
        
    def test_consecutive_moves_bug_fix(self):
        """
        Regression test for the bug where the 3rd move was not detected.
        Ensures state is correctly reset after each move.
        """
        detector = StableMoveDetector(stability_threshold=3)
        
        # 0. Initial State
        fen_0 = "start_fen"
        # 1. Move 1: e2e4
        fen_1 = "fen_after_e2e4" 
        # 2. Move 2: e7e5 
        fen_2 = "fen_after_e7e5" 
        # 3. Move 3: g1f3
        fen_3 = "fen_after_g1f3"
        
        # Helper to stabilize a FEN and return result
        def stabilize(fen):
            res = None
            for _ in range(3):
                res = detector.process(fen)
            return res
            
        # --- 0. Startup ---
        stabilize(fen_0)
        self.assertEqual(detector.confirmed_fen, fen_0)
        
        # --- 1. First Move ---
        # Mock detect_move to return dummy strings since we are using dummy FENs
        # BUT calculate_move imports detect_move from move_detector. 
        # Since we can't easily mock that without patching, let's use REAL FENs or just trust logic.
        # To avoid dependency on move_detector logic for this test, we can check that it TRIES to return something
        # or we use valid FENs. Let's use valid FENs to be safe.
        
        f0 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        f1 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1" # e2e4
        f2 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2" # e7e5
        f3 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2" # g1f3
        
        # Reset detector with real FENs
        detector = StableMoveDetector(stability_threshold=3)
        stabilize(f0)
        
        # Move 1
        m1 = stabilize(f1)
        self.assertEqual(m1, "e2e4")
        self.assertEqual(detector.confirmed_fen, f1)
        
        # Move 2
        m2 = stabilize(f2)
        self.assertEqual(m2, "e7e5")
        self.assertEqual(detector.confirmed_fen, f2)
        
        # Move 3 (The Bug Case)
        m3 = stabilize(f3)
        self.assertEqual(m3, "g1f3", "Third move should be detected!")
        self.assertEqual(detector.confirmed_fen, f3)

    def test_ignore_invalid_moves_but_update_state(self):
        detector = StableMoveDetector(stability_threshold=2)
        # Valid FEN strings required for parsing
        fen_start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        # Fen teleport: piece moves from a1 to a3 but also h1 to h3 (invalid double move)
        fen_double = "rnbqkbnr/pppppppp/8/8/8/P6P/1PPPPPP1/RNBQKBNR b KQkq - 0 1"
        
        # 1. Start
        detector.process(fen_start)
        detector.process(fen_start)
        
        # 2. Invalid Move (Double move)
        detector.process(fen_double)
        move = detector.process(fen_double)
        
        # Should return None because detect_move rejects >2 changes
        self.assertIsNone(move)
        # But state should be updated
        self.assertEqual(detector.confirmed_fen, fen_double)

if __name__ == '__main__':
    unittest.main()
