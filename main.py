import cv2
import numpy as np
import cvzone
# from piece_recognition import PieceRecognizer # REMOVED
from state_tracker import StateTracker
import board_detection
import fen_generator
from grid_extractor import GridExtractor
from square_classifier import SquareClassifier

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CAMERA_ID = 0
WIDTH, HEIGHT = 1280, 720
DISPLAY_SIZE = (1280, 720)
BOARD_MARGIN = 100
CROP_OFFSET = 0

# ---------------------------------------------------------------------------
# Globals for Mouse Callback
# ---------------------------------------------------------------------------
mouse_start = (-1, -1)
mouse_current = (-1, -1)
is_drawing = False
roi_selected = False
selection_rect = None

def mouse_callback(event, x, y, flags, param):
    global mouse_start, mouse_current, is_drawing, roi_selected, selection_rect
    
    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True
        roi_selected = False
        mouse_start = (x, y)
        mouse_current = (x, y)
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing:
            # We don't need to "draw" here, just update current pos
            mouse_current = (x, y)
            
    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False
        roi_selected = True
        mouse_current = (x, y)
        # Normalize rect (handle drawing backwards)
        x1, y1 = mouse_start
        x2, y2 = mouse_current
        # Ensure we have a valid rect even if just a click
        if abs(x1-x2) > 5 and abs(y1-y2) > 5:
            selection_rect = (min(x1, x2), min(y1, y2), abs(x1-x2), abs(y1-y2))
        else:
            roi_selected = False
            selection_rect = None
            print("selecao muito pequena, ignorada.")


def main():
    global roi_selected, is_drawing, selection_rect, mouse_current, mouse_start
    
    # Initialize components
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)
    
    grid_extractor = GridExtractor()
    classifier = SquareClassifier()
    tracker = StateTracker(history_length=5)
    
    board_detection_mode = True
    board_corners = None
    rotation_state = 0
    debug_mode = False
    
    print("iniciando sistema de visao de xadrez (GRID-BASED)...")
    print("PASSO 1: CONFIGURAÇÃO INICIAL")
    print("   Certifique-se que o tabuleiro está na POSIÇÃO INICIAL (todas as peças).")
    print("   1. Desenhe o retangulo.")
    print("   2. Pressione 'c' para confirmar e CALIBRAR.")
    print("   3. O sistema aprendera a aparencia das pecas.")
    print("   4. Pressione 'r' para RESETAR se precisar.")
    print("   5. Pressione 'q' para sair.")
    
    window_name = "visao de xadrez"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while True:
        success, img = cap.read()
        if not success:
            break
            
        img_display = img.copy()
        
        if board_detection_mode:
            # -------------------------------------------------------
            # 1. Manual Selection Mode
            # -------------------------------------------------------
            # Draw temporary box while dragging
            if is_drawing:
                cv2.rectangle(img_display, mouse_start, mouse_current, (0, 255, 255), 2)
            
            # Show fixed selection if made
            if roi_selected and selection_rect is not None:
                x, y, w, h = selection_rect
                cv2.rectangle(img_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cvzone.putTextRect(img_display, "tabuleiro selecionado! certifique-se que esta na posicao inicial!", (50, 20), scale=1.5, thickness=2, colorR=(0,0,0), colorT=(0,255,255), offset=10)
                cvzone.putTextRect(img_display, "'c' confirmar e CALIBRAR", (50, 60), scale=2, thickness=2, colorR=(0,0,0), colorT=(0,255,0), offset=10)
                
                # Check for confirm
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    # Convert rect to corners for warp_image (TL, TR, BR, BL)
                    board_corners = np.array([
                        [[x, y]],
                        [[x+w, y]],
                        [[x+w, y+h]],
                        [[x, y+h]]
                    ], dtype=np.int32)
                    
                    # ---------------------------------------
                    # CALIBRATION SEQUENCE
                    # ---------------------------------------
                    print("Calibrando...")
                    # 1. Warp current frame - using board_detection logic
                    # Ensure corners are reshaped correctly for reorder
                    corners_reshaped = board_corners.reshape(4, 2)
                    points_ordered = board_detection.reorder(corners_reshaped)
                    img_warped, matrix, board_size = board_detection.warp_image(img, points_ordered)
                    
                    # 2. Extract Squares
                    squares = grid_extractor.split_board(img_warped)
                    
                    # 3. Train Classifier
                    classifier.train(squares)
                    
                    board_detection_mode = False
                    rotation_state = 0
                    print("Sistema Calibrado! Pode mover as pecas.")
                    
                elif key == ord('r'):
                    roi_selected = False
                    selection_rect = None
                    print("selecao resetada.")
            
            else:
                 # No selection yet
                 cvzone.putTextRect(img_display, "desenhe o retangulo no tabuleiro", (50, 50), scale=2, thickness=2, colorR=(0, 0, 255), offset=10)

            cv2.imshow(window_name, img_display)

        else:
            # -------------------------------------------------------
            # 2. Gameplay Mode (Grid Classification)
            # -------------------------------------------------------
            # Recalculate warp for current frame (using same corners)
            corners_reshaped = board_corners.reshape(4, 2)
            points_ordered = board_detection.reorder(corners_reshaped)
            img_warped, matrix, board_size = board_detection.warp_image(img, points_ordered)
            
            # Draw Grid lines on visualization
            img_vis = img_warped.copy()
            img_vis = board_detection.draw_chess_grid(img_vis, board_size)
            
            # Extract current squares
            squares = grid_extractor.split_board(img_warped)
            
            # Classify and Build Map
            board_map = {}
            for (f, r), sq_img in squares.items():
                label = classifier.predict(sq_img)
                
                # Only add if it's a piece (not None/Empty) for visuals, 
                # but for FEN we might need to know emptiness?
                # Actually fen_generator usually iterates 8x8 and checks the map.
                # If key missing -> empty.
                if label and label != 'empty':
                    board_map[(f, r)] = {'fen': label}
                    
                    # Draw on Vis
                    square_size = board_size // 8
                    # Visual coords (f=0 is left/a, r=0 is bottom/1 in logic)
                    # GridExtractor: (0,0) is A1 (Bottom Left visually if Rank 1 is bottom)
                    # BUT GridExtractor loop: 
                    # r=0 (Top/Rank8), c=0 (Left/FileA) -> logical (0, 7)
                    # Wait, GridExtractor implementation:
                    # logical_file = c
                    # logical_rank = 7 - r
                    # So pixel (0,0) is (File 0, Rank 7) -> A8. Correct.
                    
                    # Visual Drawing:
                    # We want to draw at pixel corresponding to (f, r).
                    # r is logical rank index (0=Rank1, 7=Rank8).
                    # Visual row index (0=Top, 7=Bottom) = 7 - r.
                    # f is logical file index (0=A, 7=H).
                    # Visual col index = f.
                    
                    vis_row = 7 - r
                    vis_col = f
                    
                    px = int(vis_col * square_size + square_size / 2)
                    py = int(vis_row * square_size + square_size / 2)
                    
                    # Color based on piece case?
                    color = (0, 0, 255) # Red for all for now, or distinguish
                    if label.isupper(): # White
                        color = (255, 255, 255)
                    else: # Black
                        color = (0, 0, 0)
                        
                    cv2.putText(img_vis, label, (px-10, py+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Generate FEN
            current_fen = fen_generator.generate_fen(board_map)
            
            # Update State
            stable_fen, changed = tracker.update(current_fen)
            
            # Display Status
            status_text = f"FEN: {stable_fen if stable_fen else 'estabilizando...'}"
            cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
            
            if changed:
                print(f"\n[movimento detectado] novo fen: {stable_fen}")
                
            # Show images
            cv2.imshow("visao de xadrez - principal", img)
            cv2.imshow("visao de xadrez - processado", img_vis)

        # Key Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
             board_detection_mode = True
             roi_selected = False
             selection_rect = None
             print("RESET: Re-calibrar.")
             try:
                 cv2.destroyWindow("visao de xadrez - processado")
             except:
                 pass
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()