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
