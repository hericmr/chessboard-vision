import cv2
import numpy as np
from piece_detector import PieceDetector

class ChangeDetector:
    def __init__(self):
        self.z_threshold = 2.5
        self.initial_variance = 100
        self.alpha = 0.1
        self.blur_kernel = 5
        self._kernel = 5
        
        self.means = {}
        self.variances = {}
        self.is_calibrated = False
        self.focus_squares = set() # Optional: limit detection to these squares
        
        self.piece_detector = PieceDetector() # Helper for circularity

    def calibrate(self, squares):
        """Initialize background model (Mean/Variance) from current frame."""
        self.means = {}
        self.variances = {}
        is_float = True
        
        for pos, img in squares.items():
            gray = self._preprocess(img)
            self.means[pos] = gray.astype(np.float32)
            self.variances[pos] = np.full(gray.shape, self.initial_variance, dtype=np.float32)
            
        self.is_calibrated = True

    def _preprocess(self, img):
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        k = self.blur_kernel | 1 # Ensure odd
        return cv2.GaussianBlur(gray, (k, k), 0)

    def set_focus_squares(self, squares):
        self.focus_squares = set(squares)

    def clear_focus(self):
        self.focus_squares = set()

    def get_focus_count(self):
        return len(self.focus_squares) if self.focus_squares else 64

    def update_all_references(self, squares):
        """Update background model (Mean/Variance) using Alpha."""
        if not self.is_calibrated:
            self.calibrate(squares)
            return

        for pos, img in squares.items():
            if self.focus_squares and pos not in self.focus_squares:
                continue

            gray = self._preprocess(img).astype(np.float32)
            mean = self.means[pos]
            var = self.variances[pos]
            
            # Update Mean: (1-alpha)*prev + alpha*curr
            new_mean = (1 - self.alpha) * mean + self.alpha * gray
            
            # Update Variance: (1-alpha)*prev + alpha*(diff^2)
            diff = gray - new_mean
            new_var = (1 - self.alpha) * var + self.alpha * (diff ** 2)
            
            # Constraint variance to avoid 0
            new_var = np.maximum(new_var, 10.0)
            
            self.means[pos] = new_mean
            self.variances[pos] = new_var

    def detect_changes(self, squares):
        """Returns valid changes (dict of square -> magnitude)."""
        detailed = self.detect_changes_detailed(squares)
        # Filter for significant changes
        # For simple detection, return anything that is 'PARCIAL' or 'TOTAL'
        changes = {}
        for pos, info in detailed.items():
            if info['intensity'] in ['PARCIAL', 'TOTAL']:
                changes[pos] = info['pct_changed']
        return changes

    def detect_changes_detailed(self, squares):
        """
        Full analysis with Z-scores and circularity check.
        Returns dict: {pos: {'z_score': float, 'pct_changed': float, 'intensity': str, 'is_circular': bool}}
        """
        results = {}
        if not self.is_calibrated:
            return results

        # Determine which squares to check
        to_check = self.focus_squares if self.focus_squares else squares.keys()

        for pos in to_check:
            if pos not in squares: continue
            
            img = squares[pos]
            gray = self._preprocess(img).astype(np.float32)
            
            mean = self.means.get(pos)
            var = self.variances.get(pos)
            
            if mean is None: continue
            
            # 1. Z-Score Calculation
            # Z = |Current - Mean| / StdDev
            # Change if Z > Threshold
            std_dev = np.sqrt(var)
            diff = np.abs(gray - mean)
            z_score_map = diff / std_dev
            
            # Mask of changed pixels
            changed_mask = z_score_map > self.z_threshold
            changed_pixels = np.count_nonzero(changed_mask)
            total_pixels = gray.size
            pct_changed = (changed_pixels / total_pixels) * 100
            
            if pct_changed < 5.0: # Ignore negligible noise
                continue
                
            # 2. Intensity Classification
            if pct_changed > 75:
                intensity = 'TOTAL'
            elif pct_changed > 15:
                intensity = 'PARCIAL'
            else:
                intensity = 'LEVE' # Light changes (shadows)
            
            # 3. Circularity Check (for Parcial/Moved pieces)
            # Use PieceDetector on the CURRENT image to see if there's a piece
            # OR check if the *difference* looks like a piece?
            # Usually we check if the current view looks like a piece.
            pd_result = self.piece_detector.detect_piece(img, pos)
            is_circular = pd_result['has_piece']
            
            results[pos] = {
                'z_score': float(np.max(z_score_map)), # Peak Z-score
                'pct_changed': pct_changed,
                'intensity': intensity,
                'is_circular': is_circular,
                'center_ratio': 1.0 # placeholder
            }
            
        return results

    def classify_hand_pattern(self, detailed):
        """
        Determine if the set of changes looks like a Hand or a Move.
        return {'is_hand': bool, 'is_move': bool, 'move_candidates': set(pos)}
        """
        total_squares = len(detailed)
        total_intensity = sum(1 for v in detailed.values() if v['intensity'] == 'TOTAL')
        parcial_squares = [pos for pos, v in detailed.items() if v['intensity'] == 'PARCIAL']
        
        # Heuristic 1: If too many squares blocked TOTAL -> Hand
        if total_intensity >= 2 or total_squares >= 4:
            return {'is_hand': True, 'is_move': False, 'move_candidates': set()}
            
        # Heuristic 2: If we have a connected cluster of changes -> Hand
        # (Simplified: just count for now)
        if total_squares > 2:
             return {'is_hand': True, 'is_move': False, 'move_candidates': set()}
             
        # Heuristic 3: Move Candidate
        # Typically 2 squares change for a move (From->To)
        # Or 1 square if lifted (From)
        move_candidates = set(detailed.keys())
        
        # Refine: If a square changed 'TOTAL' it's likely the hand covering it
        # If 'PARCIAL' and 'is_circular', it's likely the piece landing or lifting?
        # Actually, when lifting, detecting 'is_circular' might fail if empty?
        # When landing, 'is_circular' should be true.
        
        return {
            'is_hand': False, 
            'is_move': (len(move_candidates) == 2), 
            'move_candidates': move_candidates
        }
