import cv2
import numpy as np
import time
import json
import os
cimport numpy as np

class ImageEnhancerCython:
    """
    Cython version of ImageEnhancer.
    A modular pipeline to improve visual quality of webcam frames.
    Implements lighting correction, noise reduction, sharpening, and normalization.
    """
    def __init__(self, float clahe_clip_limit=3.0, tuple tile_grid_size=(8, 8)):
        """
        Initialize the enhancer with configurable parameters.
        :param clahe_clip_limit: Threshold for contrast limiting in CLAHE.
        :param tile_grid_size: Size of grid for histogram equalization.
        """
        # CLAHE (Contrast Limited Adaptive Histogram Equalization) setup
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=tile_grid_size)
        
        # Sharpening kernel
        self.sharpen_kernel = np.array([[-1, -1, -1],
                                        [-1,  9, -1],
                                        [-1, -1, -1]], dtype=np.float32)
        
        self.profile = self.load_profile()

    def load_profile(self):
        try:
            if os.path.exists("color_profile.json"):
                print("Loaded color profile (Cython)")
                with open("color_profile.json", "r") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading profile: {e}")
        return {}

    def apply_color_profile(self, frame):
        if not self.profile:
            return frame
            
        # Extract parameters with defaults
        cdef float hue_shift = self.profile.get("hue_shift", 0.0)
        cdef float sat_scale = self.profile.get("sat_scale", 1.0)
        cdef float val_scale = self.profile.get("val_scale", 1.0)
        cdef float contrast = self.profile.get("contrast", 1.0)
        cdef int brightness = self.profile.get("brightness", 0)
        cdef int radical_mode = self.profile.get("radical_mode", 0)
        cdef float target_hue = self.profile.get("target_hue", 0.0)
        cdef float hue_window = self.profile.get("hue_window", 20.0)

        # 1. Contrast/Brightness
        frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
        
        # 2. HSV Adjustments
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        
        if radical_mode:
            h_dist = np.abs(h - target_hue)
            h_dist = np.minimum(h_dist, 180 - h_dist)
            mask = h_dist < hue_window
            
            # Boost target saturation, desaturate others
            # Using numpy operations which are optimized in C if types are clear
            s[mask] = s[mask] * 2.0 
            s[~mask] = s[~mask] * 0.5
            
        # Apply global adjustments
        h = (h + hue_shift) % 180
        s = s * sat_scale
        v = v * val_scale
        
        # Clip values
        h = np.clip(h, 0, 179)
        s = np.clip(s, 0, 255)
        v = np.clip(v, 0, 255)
        
        hsv_final = cv2.merge([h, s, v])
        hsv_final = hsv_final.astype(np.uint8)
        
        return cv2.cvtColor(hsv_final, cv2.COLOR_HSV2BGR)

    def correct_lighting(self, frame):
        """
        1. Lighting Correction
        Converts to LAB color space and applies CLAHE to the L (Lightness) channel.
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
        Uses Bilateral Filter.
        """
        return cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

    def sharpen(self, frame):
        """
        3. Sharpening
        """
        return cv2.filter2D(frame, -1, self.sharpen_kernel)

    def normalize_intensity(self, frame):
        """
        4. Normalization
        """
        return cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    def prepare_analysis(self, frame):
        """
        5. Preparation for Analysis
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(gray_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return gray, binary

    def process_pipeline(self, frame):
        """
        Executes the full pipeline sequentially.
        """
        # Step 0: Color Profile
        frame = self.apply_color_profile(frame)

        # Step 1: Lighting
        enhanced = self.correct_lighting(frame)
        
        # Step 2: Noise
        enhanced = self.reduce_noise(enhanced)
        
        # Step 3: Sharpening
        enhanced = self.sharpen(enhanced)
        
        # Step 4: Normalization
        enhanced = self.normalize_intensity(enhanced)
        
        return enhanced
