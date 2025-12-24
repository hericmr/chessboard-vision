"""
Tests for NoiseHandler state machine.
"""

import unittest
from noise_handler import NoiseHandler, NoiseState


class TestNoiseHandler(unittest.TestCase):
    def setUp(self):
        self.handler = NoiseHandler()
    
    def test_initial_state_is_idle(self):
        self.assertEqual(self.handler.state, NoiseState.IDLE)
        self.assertFalse(self.handler.is_blocked())
    
    def test_no_changes_stays_idle(self):
        state, data = self.handler.process(set())
        self.assertEqual(state, NoiseState.IDLE)
        self.assertEqual(data["message"], "waiting")
    
    def test_single_change_triggers_pending(self):
        state, data = self.handler.process({(4, 1)})
        self.assertEqual(state, NoiseState.MOVE_PENDING)
        self.assertEqual(data["lifted"], (4, 1))
        self.assertFalse(data["stable"])
    
    def test_many_changes_triggers_noise(self):
        # More than 3 changes = noise
        changes = {(0, 0), (1, 0), (2, 0), (3, 0)}
        state, data = self.handler.process(changes)
        self.assertEqual(state, NoiseState.NOISE_ACTIVE)
        self.assertTrue(self.handler.is_blocked())
    
    def test_stability_counter(self):
        # Start with 2 changes
        changes = {(4, 1), (4, 3)}
        
        # Process same changes multiple times
        for i in range(NoiseHandler.STABILITY_FRAMES - 1):
            state, data = self.handler.process(changes)
            self.assertEqual(state, NoiseState.MOVE_PENDING)
            self.assertFalse(data["stable"])
        
        # Final frame should be stable
        state, data = self.handler.process(changes)
        self.assertTrue(data["stable"])
    
    def test_noise_clears_after_cooldown(self):
        # Enter noise state
        changes = {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)}
        self.handler.process(changes)
        self.assertEqual(self.handler.state, NoiseState.NOISE_ACTIVE)
        
        # Clear with empty frames
        for i in range(NoiseHandler.COOLDOWN_FRAMES):
            state, data = self.handler.process(set())
        
        self.assertEqual(state, NoiseState.IDLE)
        self.assertFalse(self.handler.is_blocked())
    
    def test_reset(self):
        # Get into a state
        self.handler.process({(4, 1), (4, 3)})
        self.handler.reset()
        
        self.assertEqual(self.handler.state, NoiseState.IDLE)
        self.assertEqual(self.handler.stable_count, 0)
        self.assertEqual(len(self.handler.pending_squares), 0)
    
    def test_get_state_name(self):
        self.assertEqual(self.handler.get_state_name(), "IDLE")
        
        self.handler.process({(0, 0), (1, 0), (2, 0), (3, 0)})
        self.assertEqual(self.handler.get_state_name(), "NOISE")
        
        self.handler.reset()
        self.handler.process({(4, 1)})
        self.assertEqual(self.handler.get_state_name(), "PENDING")


if __name__ == '__main__':
    unittest.main()
