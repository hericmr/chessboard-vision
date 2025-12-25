
import cv2
import numpy as np
import json
import os

PROFILE_FILE = "color_profile.json"

def nothing(x):
    pass

class ColorCalibrator:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.window_name = "Color Calibration"
        cv2.namedWindow(self.window_name)
        
        # Load existing profile or defaults
        self.profile = self.load_profile()
        
        # Create Trackbars
        # Hue Shift: -180 to 180 (OpenCV wraps 0-179, so we'll map 0-360 to shift)
        cv2.createTrackbar("Hue Shift", self.window_name, self.profile.get("hue_shift", 0) + 180, 360, nothing)
        
        # Saturation Scale: 0.0 to 3.0 (mapped from 0-300)
        cv2.createTrackbar("Sat Boost", self.window_name, int(self.profile.get("sat_scale", 1.0) * 100), 300, nothing)
        
        # Value/Brightness Scale: 0.0 to 3.0 (mapped from 0-300)
        cv2.createTrackbar("Val Boost", self.window_name, int(self.profile.get("val_scale", 1.0) * 100), 300, nothing)
        
        # Contrast: 0.5 to 3.0 (mapped from 0-250, offset by 50)
        # We'll use a simple linear contrast: new = alpha * old + beta
        cv2.createTrackbar("Contrast", self.window_name, int((self.profile.get("contrast", 1.0) * 100)), 300, nothing)
        
        # Brightness (Beta): -100 to 100 (mapped from 0-200)
        cv2.createTrackbar("Brightness", self.window_name, self.profile.get("brightness", 0) + 100, 200, nothing)

        # Radical Isolation Mode (Lilac/White separation)
        # Isolate specific hue range and boost/suppress others
        cv2.createTrackbar("Radical Mode", self.window_name, self.profile.get("radical_mode", 0), 1, nothing)
        cv2.createTrackbar("Target Hue", self.window_name, self.profile.get("target_hue", 0), 179, nothing) # Target Lilac hue
        cv2.createTrackbar("Hue Window", self.window_name, self.profile.get("hue_window", 20), 50, nothing)

    def load_profile(self):
        if os.path.exists(PROFILE_FILE):
            try:
                with open(PROFILE_FILE, "r") as f:
                    print(f"Loading profile from {PROFILE_FILE}")
                    return json.load(f)
            except Exception as e:
                print(f"Error loading profile: {e}")
        return {}

    def save_profile(self, profile):
        try:
            with open(PROFILE_FILE, "w") as f:
                json.dump(profile, f, indent=4)
            print(f"Profile saved to {PROFILE_FILE}")
        except Exception as e:
            print(f"Error saving profile: {e}")

    def apply_color_adjustments(self, frame, hue_shift, sat_scale, val_scale, contrast, brightness, radical_mode, target_hue, hue_window):
        # 1. Contrast/Brightness (Linear)
        # alpha = contrast, beta = brightness
        frame_cb = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
        
        # 2. HSV Adjustments
        hsv = cv2.cvtColor(frame_cb, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        
        if radical_mode:
            # Radical Mode: Isolate specific hue (e.g., Lilac) and separate from others
            
            # Distance from target hue (handling wrap-around)
            h_dist = np.abs(h - target_hue)
            h_dist = np.minimum(h_dist, 180 - h_dist)
            
            # Create mask for target color
            # If inside window => retain or boost. If outside => suppress or invert.
            # Strategy: Make target hue darker/lighter, everything else opposite.
            
            # Simple "Spotlight" effect: boost saturation of target, desaturate others
            mask = h_dist < hue_window
            
            # Boost target saturation significantly to distinguish from white
            s[mask] = s[mask] * 2.0 
            
            # Desaturate others (beiges/whites will lose slight color)
            s[~mask] = s[~mask] * 0.5
            
            # Radical contrast: Darken target color, lighten others?
            # Or make target color POP.
            # Let's try boosting Value for non-target (white/beige) and keeping target normal
            # v[~mask] += 30
            pass

        # Apply global Hue Shift
        h = (h + hue_shift) % 180
        
        # Apply Global Saturation/Value Scales
        s = s * sat_scale
        v = v * val_scale
        
        # Clip values
        h = np.clip(h, 0, 179)
        s = np.clip(s, 0, 255)
        v = np.clip(v, 0, 255)
        
        # Merge and convert back
        hsv_final = cv2.merge([h, s, v])
        hsv_final = hsv_final.astype(np.uint8)
        
        return cv2.cvtColor(hsv_final, cv2.COLOR_HSV2BGR)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x, self.mouse_y = x, y

    def run(self):
        print("Controls:")
        print("  - Adjust trackbars to separate PIECES from BOARD.")
        print("  - Look at the GRAYSCALE view - this is what the AI sees.")
        print("  - Press 's' to SAVE profile.")
        print("  - Press 'q' to QUIT.")
        
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.mouse_x, self.mouse_y = 0, 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame = cv2.resize(frame, (640, 480))
            
            # Read Trackbars
            hue_shift = cv2.getTrackbarPos("Hue Shift", self.window_name) - 180
            sat_scale = cv2.getTrackbarPos("Sat Boost", self.window_name) / 100.0
            val_scale = cv2.getTrackbarPos("Val Boost", self.window_name) / 100.0
            contrast = cv2.getTrackbarPos("Contrast", self.window_name) / 100.0
            brightness = cv2.getTrackbarPos("Brightness", self.window_name) - 100
            
            radical_mode = cv2.getTrackbarPos("Radical Mode", self.window_name)
            target_hue = cv2.getTrackbarPos("Target Hue", self.window_name)
            hue_window = cv2.getTrackbarPos("Hue Window", self.window_name)
            
            # Apply processing
            processed = self.apply_color_adjustments(frame, hue_shift, sat_scale, val_scale, contrast, brightness, radical_mode, target_hue, hue_window)
            
            # Gray view (What the computer sees)
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            # Stack images: Original | Processed | Grayscale
            combined = np.hstack([frame, processed, gray_bgr])
            
            # Add labels
            cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, "Enhanced Color", (640 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, "Computer Vision (Gray)", (1280 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show pixel value under cursor
            if 0 <= self.mouse_x < combined.shape[1] and 0 <= self.mouse_y < combined.shape[0]:
                pixel = combined[self.mouse_y, self.mouse_x]
                # Determine which section we are in
                section = "Original"
                if self.mouse_x > 1280: section = "Gray"
                elif self.mouse_x > 640: section = "Enhanced"
                
                # Get HSV value if in original/enhanced
                if section != "Gray":
                    # Extract pixel from relevant part of concatenated image is tricky? 
                    # Actually combined has BGR values.
                    b, g, r = pixel
                    # To get Hue, we need to convert just this pixel or assume from position
                    # Let's just show BGR for now, simplicity.
                    info = f"{section} BGR: {b},{g},{r}"
                else:
                    info = f"Gray Intensity: {pixel[0]}"
                
                cv2.putText(combined, info, (10, combined.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Resize to fit screen width if necessary (optional)
            scale = 0.8
            h, w = combined.shape[:2]
            final_view = cv2.resize(combined, (int(w*scale), int(h*scale)))
            
            cv2.imshow(self.window_name, final_view)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                profile = {
                    "hue_shift": hue_shift,
                    "sat_scale": sat_scale,
                    "val_scale": val_scale,
                    "contrast": contrast,
                    "brightness": brightness,
                    "radical_mode": radical_mode,
                    "target_hue": target_hue,
                    "hue_window": hue_window
                }
                self.save_profile(profile)
                
                # Visual feedback
                cv2.rectangle(final_view, (0, 0), (w, h), (0, 255, 0), 10)
                cv2.imshow(self.window_name, final_view)
                cv2.waitKey(200)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ColorCalibrator().run()
