"""
Ultra-Simple Change Detector

Detects CHANGES from a reference state, not absolute occupied/empty.
Loads settings from sensitivity_settings.json.
"""

import cv2
import numpy as np
import json
import os

SETTINGS_FILE = "sensitivity_settings.json"
DEFAULT_SETTINGS = {
    "sensitivity": 25,
    "blur_kernel": 5,
    "stable_frames": 15,
}


class ChangeDetector:
    """Detects CHANGES in squares from a reference state."""
    
    def __init__(self):
        """Load settings from file or use defaults."""
        self.settings = self._load_settings()
        self.sensitivity = self.settings["sensitivity"]
        self.blur_kernel = self.settings["blur_kernel"]
        self.stable_frames = self.settings["stable_frames"]
        self.reference_squares = {}
        self.is_calibrated = False
        print(f"[ChangeDetector] Sens: {self.sensitivity}, Blur: {self.blur_kernel}, Frames: {self.stable_frames}")
    
    def _load_settings(self):
        """Load all settings from file."""
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, 'r') as f:
                    data = json.load(f)
                    # Merge with defaults
                    for key in DEFAULT_SETTINGS:
                        if key not in data:
                            data[key] = DEFAULT_SETTINGS[key]
                    return data
            except:
                pass
        return DEFAULT_SETTINGS.copy()
    
    def calibrate(self, squares_dict):
        """Save current state as reference."""
        self.reference_squares = {}
        for pos, img in squares_dict.items():
            self.reference_squares[pos] = img.copy()
        self.is_calibrated = True
        print(f"[ChangeDetector] Referência capturada ({len(self.reference_squares)} casas)")
    
    def _calculate_difference(self, img1, img2):
        """Calculate mean absolute difference between two images."""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Use blur from settings
        k = max(1, self.blur_kernel | 1)
        gray1 = cv2.GaussianBlur(gray1, (k, k), 0)
        gray2 = cv2.GaussianBlur(gray2, (k, k), 0)
        
        diff = cv2.absdiff(gray1, gray2)
        return np.mean(diff)

    
    def detect_changes(self, squares_dict):
        """
        Find which squares changed from the reference.
        
        Returns:
            dict: {(f,r): diff_value} for squares that changed
        """
        if not self.is_calibrated:
            return {}
        
        changed = {}
        
        for pos, current_img in squares_dict.items():
            if pos not in self.reference_squares:
                continue
            
            ref_img = self.reference_squares[pos]
            diff = self._calculate_difference(current_img, ref_img)
            
            if diff > self.sensitivity:
                changed[pos] = diff
        
        return changed
    
    def update_reference(self, pos, img):
        """Update reference for a single square."""
        self.reference_squares[pos] = img.copy()
    
    def update_all_references(self, squares_dict):
        """Update all references (after confirmed move)."""
        for pos, img in squares_dict.items():
            self.reference_squares[pos] = img.copy()
    
    # Compatibility methods
    def train(self, squares_dict):
        """Alias for calibrate."""
        self.calibrate(squares_dict)
    
    def predict(self, square_img, pos=None):
        """
        For compatibility - returns if square changed from reference.
        """
        if not self.is_calibrated or pos is None or pos not in self.reference_squares:
            return {'label': 'unknown', 'occupied': False, 'changed': False}
        
        ref_img = self.reference_squares[pos]
        diff = self._calculate_difference(square_img, ref_img)
        
        changed = diff > self.sensitivity
        
        return {
            'label': 'changed' if changed else 'unchanged',
            'occupied': None,  # We don't know absolute state
            'changed': changed,
            'diff': diff
        }
    
    def update_background(self, pos, square_img):
        """Alias for update_reference."""
        self.update_reference(pos, square_img)
