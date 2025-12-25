
import sys
import os
sys.path.append(os.getcwd())

try:
    import frame_enhancer
    print(f"Imported frame_enhancer successfully")
    print(f"ImageEnhancer class: {frame_enhancer.ImageEnhancer}")
    
    if "Cython" in str(frame_enhancer.ImageEnhancer) or "cython" in str(frame_enhancer.ImageEnhancer).lower():
        print("SUCCESS: Using Cython implementation as expected.")
    elif frame_enhancer.USE_CYTHON:
        # If it's a python class but USE_CYTHON is True, check if it's the wrapper or the fallback
        print("INFO: Checking class hierarchy...")
    
    enhancer = frame_enhancer.ImageEnhancer()
    print("Enhancer instantiated successfully")
    
except Exception as e:
    print(f"TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
