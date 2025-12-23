import cv2
import numpy as np

class SquareClassifier:
    def __init__(self):
        self.templates = {} # Map 'fen_char' -> list of template images
        # Standard Initial Board State
        # Rank 1 (White pieces)
        # Rank 2 (White Pawns)
        # Rank 7 (Black Pawns)
        # Rank 8 (Black Pieces)
        # Empty: Rows 2-5 (indices)
        self.initial_setup = {
            (0, 0): 'R', (1, 0): 'N', (2, 0): 'B', (3, 0): 'Q', (4, 0): 'K', (5, 0): 'B', (6, 0): 'N', (7, 0): 'R', # Rank 1 (White)
            (0, 1): 'P', (1, 1): 'P', (2, 1): 'P', (3, 1): 'P', (4, 1): 'P', (5, 1): 'P', (6, 1): 'P', (7, 1): 'P', # Rank 2
            (0, 6): 'p', (1, 6): 'p', (2, 6): 'p', (3, 6): 'p', (4, 6): 'p', (5, 6): 'p', (6, 6): 'p', (7, 6): 'p', # Rank 7 (Black)
            (0, 7): 'r', (1, 7): 'n', (2, 7): 'b', (3, 7): 'q', (4, 7): 'k', (5, 7): 'b', (6, 7): 'n', (7, 7): 'r', # Rank 8
        }
        self.is_calibrated = False

    def train(self, squares_dict):
        """
        Extract templates from the initial board state.
        squares_dict: result from GridExtractor.split_board(initial_frame)
        """
        self.templates = {}
        
        # We need a bucket for 'empty' squares
        self.templates['empty'] = []
        
        for (f, r), square_img in squares_dict.items():
            # Preprocess (Gray + Smooth)
            gray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
            # Maybe some blur
            blur = cv2.GaussianBlur(gray, (3,3), 0)
            
            # Check if this square has a starting piece
            if (f, r) in self.initial_setup:
                piece_code = self.initial_setup[(f, r)]
                
                # Initialize list if new
                if piece_code not in self.templates:
                    self.templates[piece_code] = []
                
                self.templates[piece_code].append(blur)
            
            # If it's a known empty square (Ranks 3,4,5,6 -> indices 2,3,4,5)
            elif 2 <= r <= 5:
                self.templates['empty'].append(blur)
                
        self.is_calibrated = True
        print(f"Calibrated with {len(self.templates)} classes.")

    def predict(self, square_img):
        """
        Classifies a single square image against learnt templates.
        Returns: Piece Code (e.g. 'P', 'k') or None (Empty).
        """
        if not self.is_calibrated:
            return None
            
        gray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        
        best_score = float('inf')
        best_label = None # None means empty
        
        # Compare against all templates
        # Using MSE (Mean Squared Error) because exact match is expected
        # Or Correlation Coefficient (TM_CCOEFF_NORMED)
        
        # Let's use simple MSE for speed and robustness on static cam
        for label, templates in self.templates.items():
            for tmpl in templates:
                # Resize template to match current square if needed (should be same size if warp is consistent)
                if tmpl.shape != blur.shape:
                    tmpl = cv2.resize(tmpl, (blur.shape[1], blur.shape[0]))
                
                # Difference
                diff = cv2.absdiff(blur, tmpl)
                score = np.sum(diff) # Sum of absolute differences (L1 norm)
                
                if score < best_score:
                    best_score = score
                    best_label = label
                    
        # Return label (None if we decide 'empty' matches 'None', or return '1' for FEN empty)
        # FEN uses numbers for empty, but our map expects None or Label?
        # Let's return the label string ('P', 'k', 'empty')
        return best_label
