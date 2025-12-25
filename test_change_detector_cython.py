
import sys
import os
sys.path.append(os.getcwd())

try:
    import change_detector
    print(f"Imported change_detector successfully")
    
    # Check what class is being used
    cd_class = change_detector.ChangeDetector
    print(f"ChangeDetector class: {cd_class}")
    
    # Instantiate
    cd = cd_class()
    print("ChangeDetector instantiated successfully")
    
    # Check for Cython-specific attributes or Type
    if "Cython" in str(cd_class) or "cython" in str(cd_class).lower():
        print("SUCCESS: Using Cython implementation as expected.")
    elif change_detector.USE_CYTHON:
        print("INFO: USE_CYTHON is True, but class name might not reflect it explicitly if aliased differently.")
        # Additional check if it's the class from the .so
        if "src.cython.change_detector_cython" in str(cd_class):
             print("SUCCESS: Confirmed Cython module source.")
        else:
             print("WARNING: Might be using Python implementation fallback.")

except Exception as e:
    print(f"TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
