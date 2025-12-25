
import unittest
import numpy as np
import cv2
import sys
import os

sys.path.append(os.getcwd())
from change_detector import ChangeDetector

class TestChangeDetectorRegression(unittest.TestCase):
    def setUp(self):
        self.detector = ChangeDetector()
        
    def test_initialization(self):
        self.assertIsNotNone(self.detector)
        print(f"Testing instance of: {type(self.detector)}")
        
    def test_calibration(self):
        # Create dummy squares
        squares = {}
        for r in range(8):
            for c in range(8):
                # Random noise image
                img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
                squares[(c, r)] = img
                
        self.detector.calibrate(squares)
        self.assertTrue(self.detector.is_calibrated)
        
    def test_detection_flow(self):
        # 1. Calibrate
        squares = {}
        for r in range(8):
            for c in range(8):
                img = np.zeros((50, 50), dtype=np.uint8)
                squares[(c, r)] = img
        self.detector.calibrate(squares)
        
        # 2. Change one square significantly
        squares[(3, 3)] = np.full((50, 50), 255, dtype=np.uint8)
        
        # 3. Detect
        changes = self.detector.detect_changes(squares)
        
        # Should detect change at (3, 3)
        self.assertIn((3, 3), changes)
        # Should be > 0
        self.assertGreater(changes[(3, 3)], 50.0)
        
        # 4. Detailed detection
        detailed = self.detector.detect_changes_detailed(squares)
        self.assertIn((3, 3), detailed)
        self.assertEqual(detailed[(3, 3)]['intensity'], 'TOTAL')

if __name__ == '__main__':
    unittest.main()
