import cv2
import numpy as np
import time

class ImageEnhancer:
    """
    A modular pipeline to improve visual quality of webcam frames.
    Implements lighting correction, noise reduction, sharpening, and normalization.
    """
    def __init__(self, clahe_clip_limit=3.0, tile_grid_size=(8, 8)):
        """
        Initialize the enhancer with configurable parameters.
        :param clahe_clip_limit: Threshold for contrast limiting in CLAHE.
        :param tile_grid_size: Size of grid for histogram equalization.
        """
        # CLAHE (Contrast Limited Adaptive Histogram Equalization) setup
        # Best for local contrast enhancement without amplifying noise too much
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=tile_grid_size)
        
        # Sharpening kernel
        # Highlights edges by subtracting neighboring pixels from the center
        self.sharpen_kernel = np.array([[-1, -1, -1],
                                        [-1,  9, -1],
                                        [-1, -1, -1]])

    def correct_lighting(self, frame):
        """
        1. Lighting Correction
        Converts to LAB color space and applies CLAHE to the L (Lightness) channel.
        This handles shadows and uneven lighting better than global equalization.
        """
        # Convert BGR to LAB
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L-channel
        l_enhanced = self.clahe.apply(l)
        
        # Merge channels back
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        
        # Convert back to BGR
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def reduce_noise(self, frame):
        """
        2. Noise Reduction
        Uses Bilateral Filter to smooth textures while preserving edges.
        More expensive than Gaussian, but critical for keeping shapes sharp.
        """
        # d=9: Diameter of each pixel neighborhood
        # sigmaColor=75: Filter sigma in the color space (larger = more smoothing)
        # sigmaSpace=75: Filter sigma in the coordinate space (larger = further pixels mix)
        return cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

    def sharpen(self, frame):
        """
        3. Sharpening
        Applies a convolution kernel to enhance edges.
        """
        return cv2.filter2D(frame, -1, self.sharpen_kernel)

    def normalize_intensity(self, frame):
        """
        4. Normalization
        Normalizes pixel intensity to the full 0-255 range.
        Ensures the image isn't washed out or too dark.
        """
        return cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    def prepare_analysis(self, frame):
        """
        5. Preparation for Analysis
        Returns a grayscale version and a binary thresholded version (Otsu).
        Useful for segmentation or classification tasks.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Gaussian blur before thresholding helps reduce noise spots
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Otsu's thresholding automatically finds the optimal threshold value
        _, binary = cv2.threshold(gray_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return gray, binary

    def process_pipeline(self, frame):
        """
        Executes the full pipeline sequentially.
        """
        # Step 1: Lighting
        enhanced = self.correct_lighting(frame)
        
        # Step 2: Noise (applied after lighting to avoid amplifying noise first)
        enhanced = self.reduce_noise(enhanced)
        
        # Step 3: Sharpening
        enhanced = self.sharpen(enhanced)
        
        # Step 4: Normalization
        enhanced = self.normalize_intensity(enhanced)
        
        return enhanced

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    enhancer = ImageEnhancer()
    
    print("Starting Frame Enhancer... Press 'q' to quit.")
    
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # --- PIPELINE ---
        enhanced_frame = enhancer.process_pipeline(frame)
        gray, binary = enhancer.prepare_analysis(enhanced_frame)
        # ----------------

        # Display FPS on frame
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # stack images for display (optional, resize to fit screen if needed)
        # Just showing separate windows for clarity
        cv2.imshow('Original Feed', frame)
        cv2.imshow('Enhanced Feed', enhanced_frame)
        cv2.imshow('Analysis (Otsu Binary)', binary)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
