import cv2
import numpy as np

class GridExtractor:
    def __init__(self):
        pass

    def split_board(self, img_warped):
        """
        Splits the warped board image into 64 individual squares.
        Returns a dictionary mapping (col, row) -> square_image.
        key: (0-7, 0-7) where 0,0 is A1 (bottom-left) and 7,7 is H8 (top-right).
        Note: The input image is assumed to be oriented with White at bottom (Rank 1).
        If the image is rotated, the caller must handle it or we map geometrically.
        
        Assuming img_warped is oriented such that:
        Top-Left is A8 (0,0 pixel) -> This is standard for OpenCV image matrix.
        Bottom-Right is H1.
        
        Let's standardize the output grid coordinate system:
        (0, 0) = A1
        (1, 0) = B1
        ...
        (7, 7) = H8
        
        If img_warped has standard visual orientation (White Bottom):
        - Top Row is Rank 8.
        - Bottom Row is Rank 1.
        - Left Col is File A.
        - Right Col is File H.
        """
        squares = {}
        rows, cols, _ = img_warped.shape
        square_h = rows // 8
        square_w = cols // 8
        
        for r in range(8): # 0 is Top (Rank 8), 7 is Bottom (Rank 1)
            for c in range(8): # 0 is Left (File A), 7 is Right (File H)
                
                # Geometry:
                y = r * square_h
                x = c * square_w
                
                # Crop with slight margin to avoid border lines if needed, or exact
                # Using exact for now
                roi = img_warped[y:y+square_h, x:x+square_w]
                
                # Logical Coordinates (0-7, 0-7) = (File, Rank_Index)
                # Rank 1 is at r=7. Rank 8 is at r=0.
                # logical_rank = 7 - r (so r=7 -> rank_idx=0 (Rank 1))
                # logical_file = c     (so c=0 -> file_idx=0 (File A))
                
                logical_file_idx = c
                logical_rank_idx = 7 - r
                
                squares[(logical_file_idx, logical_rank_idx)] = roi
                
        return squares

class SmartGridExtractor:
    def __init__(self, debug=False):
        self.grid_lines_x = None # List of 9 X-coords (0 to Width)
        self.grid_lines_y = None # List of 9 Y-coords (0 to Height)
        self.debug = debug

    def refine_grid(self, img_warped):
        """
        Analyzes the warped board to find actual internal grid lines.
        Returns (grid_x, grid_y) lists of coordinates.
        """
        h, w = img_warped.shape[:2]
        
        # 1. Edge Detection
        gray = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)
        # Use simple gradient or Canny
        edges = cv2.Canny(gray, 50, 150)
        
        # 2. Projection Profiles
        # Sum edges along rows (find horizontal lines)
        row_proj = np.sum(edges, axis=1)
        # Sum edges along cols (find vertical lines)
        col_proj = np.sum(edges, axis=0)
        
        # 3. Find peaks corresponding to grid lines
        # We expect 9 lines total (including borders) or 7 internal lines.
        # Borders are usually at 0 and W/H.
        # Let's try to find 7 internal peaks.
        
        def find_internal_lines(proj, length, count=7):
            # Split profile into 'count+1' segments effectively?
            # Or just search local maxima near expected positions.
            expected_step = length / 8.0
            lines = [0] # Start border
            
            for i in range(1, 8):
                # Search window around expected position
                expected_center = int(i * expected_step)
                search_radius = int(expected_step * 0.3)
                start = max(0, expected_center - search_radius)
                end = min(length, expected_center + search_radius)
                
                # Find max peak in this window
                window = proj[start:end]
                if len(window) > 0:
                    peak_idx = np.argmax(window)
                    actual_pos = start + peak_idx
                    lines.append(actual_pos)
                else:
                    lines.append(expected_center)
            
            lines.append(length) # End border
            return lines

        self.grid_lines_x = find_internal_lines(col_proj, w)
        self.grid_lines_y = find_internal_lines(row_proj, h)
        
        if self.debug:
            print(f"Refined X: {self.grid_lines_x}")
            print(f"Refined Y: {self.grid_lines_y}")
            
        return self.grid_lines_x, self.grid_lines_y

    def split_board(self, img_warped):
        """
        Splits board using refined grid lines if available.
        """
        if self.grid_lines_x is None or self.grid_lines_y is None:
            # Fallback to linear
            fallback = GridExtractor()
            return fallback.split_board(img_warped)
            
        squares = {}
        # Iterate files (cols) and ranks (rows)
        # grid_lines_x has indices 0..8
        # grid_lines_y has indices 0..8
        
        # Rank 8 is Top (Row 0), Rank 1 is Bottom (Row 7)
        # File A is Left (Col 0), File H is Right (Col 7)
        
        for r in range(8): # Row index 0 (Top) to 7 (Bottom)
            for c in range(8): # Col index 0 (Left) to 7 (Right)
                
                x_start = self.grid_lines_x[c]
                x_end = self.grid_lines_x[c+1]
                y_start = self.grid_lines_y[r]
                y_end = self.grid_lines_y[r+1]
                
                # Check bounds
                if x_start >= x_end or y_start >= y_end:
                    continue
                    
                roi = img_warped[y_start:y_end, x_start:x_end]
                
                # Map to Logical Coords (File, Rank)
                # r=0 -> Rank 8 (Index 7), r=7 -> Rank 1 (Index 0)
                # c=0 -> File A (Index 0)
                
                logical_file_idx = c
                logical_rank_idx = 7 - r
                
                squares[(logical_file_idx, logical_rank_idx)] = roi
                
        return squares
