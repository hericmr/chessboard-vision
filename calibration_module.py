import cv2
import numpy as np
import json
import os
import cvzone
import board_detection

class CalibrationModule:
    def __init__(self, calib_file="calibration.json", width=1280, height=720):
        self.calib_file = calib_file
        self.width = width
        self.height = height
        
        # State
        self.corners = [] # List of 4 [x, y] lists
        self.dragging_idx = -1
        self.player_color = "WHITE" # "WHITE" or "BLACK"
        # orientation_flipped: False means "Physical Board has White at Bottom" (Standard)
        #                      True means "Physical Board has Black at Bottom" (Rotated 180)
        self.orientation_flipped = False 
        
        self.window_name = "Calibracao Antigravity"
        
        self.load_calibration()

    def load_calibration(self):
        if os.path.exists(self.calib_file):
            try:
                with open(self.calib_file, "r") as f:
                    data = json.load(f)
                    raw_corners = data.get("corners", [])
                    # Sanitize: Ensure list of [x, y]
                    self.corners = []
                    for c in raw_corners:
                        c_arr = np.array(c).flatten()
                        if len(c_arr) >= 2:
                            self.corners.append([int(c_arr[0]), int(c_arr[1])])
                    
                    self.player_color = data.get("player_color", "WHITE")
                    self.orientation_flipped = data.get("orientation_flipped", False)
                    print(f"Calibration loaded: {self.player_color}, Flipped: {self.orientation_flipped}")
            except Exception as e:
                print(f"Error loading calibration: {e}")
        else:
            print("No calibration file found.")

    def save_calibration(self):
        data = {
            "corners": self.corners,
            "player_color": self.player_color,
            "orientation_flipped": self.orientation_flipped
        }
        try:
            with open(self.calib_file, "w") as f:
                json.dump(data, f)
            print("Calibration saved.")
        except Exception as e:
            print(f"Error saving calibration: {e}")

    def mouse_callback(self, event, x, y, flags, param):
        # 1. Handle Click Down (Start Dragging or adding points)
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking on an existing corner to drag
            if len(self.corners) == 4:
                for i, point in enumerate(self.corners):
                    dist = np.linalg.norm(np.array(point) - np.array([x, y]))
                    if dist < 20: # 20px radius
                        self.dragging_idx = i
                        return
            
            # If not dragging and we don't have 4 corners, add one
            if len(self.corners) < 4:
                self.corners.append([x, y])
                if len(self.corners) == 4:
                    self.sort_corners()

        # 2. Handle Mouse Move (Dragging)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_idx != -1:
                self.corners[self.dragging_idx] = [x, y]

        # 3. Handle Click Up (Stop Dragging)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging_idx != -1:
                self.dragging_idx = -1
                # Re-sort might be confusing while dragging, maybe better to only sort on explicit reset or finish?
                # But requirement says "reordenar se necessario".
                # Let's keep it manual until user confirms? or auto-sort?
                # Auto-sort ensures BL/BR/TL/TR logic is kept.
                # self.sort_corners() # Let's try NOT sorting on every drag to avoid jumping
                pass

    def sort_corners(self):
        """Use the board_detection reorder to ensure consistent order: TL, TR, BL, BR"""
        if len(self.corners) == 4:
            arr = np.array(self.corners)
            reshaped = arr.reshape((4, 1, 2)) # function expects (4, 1, 2)
            # The board_detection.reorder return format might be (4,1,2)
            # Let's check imports.
            try:
                # Assuming board_detection.reorder exists and works as observed
                reordered = board_detection.reorder(reshaped)
                # Convert back to list of [x,y]
                self.corners = [[int(p[0][0]), int(p[0][1])] for p in reordered]
            except Exception as e:
                print(f"Sort Error: {e}")

    def get_warped_preview(self, img):
        if len(self.corners) != 4:
            return None
        
        # Warp using current corners
        pts_src = np.array(self.corners, dtype=np.float32)
        
        # We need to sort them for warp perspective to work correctly with the target map
        # If we didn't sort them yet (e.g. during drag), let's sort a temporary copy
        # Actually board_detection.reorder does: TL, TR, BL, BR (Wait, check file content)
        # reorder: [0] Top-Left, [3] Bottom-Right, [1] Top-Right, [2] Bottom-Left
        # Wait, reorder lines:
        # myPointsNew[0] = Top-Left
        # myPointsNew[3] = Bottom-Right
        # myPointsNew[1] = Top-Right
        # myPointsNew[2] = Bottom-Left
        # So format is: TL, TR, BL, BR
        
        # IMPORTANT: getPerspectiveTransform expects points in a specific order usually?
        # Let's see board_detection.warp_image:
        # pts2 = [[0, 0], [width, 0], [0, height], [width, height]]
        # This matches TL, TR, BL, BR order.
        
        # So we must ensure self.corners are in TL, TR, BL, BR order.
        # Let's enforce reorder for the preview to look right.
        
        reshaped = np.array(self.corners).reshape((4, 1, 2))
        ordered_pts = board_detection.reorder(reshaped) 
        # ordered_pts is (4, 1, 2).
        # We need (4, 2) for warp_image, but warp_image in board_detection takes the complex (4,1,2) or (4,2).
        # Let's assume warp_image handles it or we pass flattened.
        
        img_warped, matrix, size = board_detection.warp_image(img, ordered_pts)
        
        # NOW: Apply Rotation Logic based on user settings
        
        # Step 1: Physical Correction
        # If orientation_flipped is True (Black at Bottom of Physical Cam),
        # but the Canonical view expects White at Bottom, we must Rotate 180
        # to get the "Canonical" board.
        if self.orientation_flipped:
            img_warped = cv2.rotate(img_warped, cv2.ROTATE_180)
            
        # At this point, img_warped is CANONICAL (White at Bottom).
        
        # Step 2: Player View
        # If Player is Black, they want to see Black at Bottom.
        # So we rotate the canonical board 180 to show them their view.
        display_warped = img_warped.copy()
        if self.player_color == "BLACK":
            display_warped = cv2.rotate(display_warped, cv2.ROTATE_180)
            
        return display_warped

    def draw_ui(self, img, img_preview):
        # 1. Draw instructions on main Image
        cvzone.putTextRect(img, "Modo Calibracao", (20, 40), scale=2, thickness=2, offset=10)
        
        info_lines = [
            f"[R] Resetar Pontos ({len(self.corners)}/4)",
            f"[F] Lado Fisico: {'PRETAS EM BAIXO' if self.orientation_flipped else 'BRANCAS EM BAIXO'}",
            f"[P] Jogador: {self.player_color}",
            "[ENTER] Confirmar e Salvar",
            "[Q] Sair"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(img, line, (20, 80 + i*30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
            
        # Draw Polygons
        if len(self.corners) > 0:
            for i, p in enumerate(self.corners):
                cv2.circle(img, (p[0], p[1]), 8, (0, 255, 255), -1)
                cv2.putText(img, str(i+1), (p[0]+10, p[1]-10), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 2)
        
        if len(self.corners) == 4:
            # Draw lines
            pts = np.array(self.corners, np.int32)
            pts = pts.reshape((-1, 1, 2))
            # We want to draw the Hull or just 0-1-3-2-0 order?
            # Since dragging allows crossing, drawing a convex hull is safer for visualization,
            # but drawing simple loop is fine.
            # Let's draw based on current list order 0-1-2-3-0
            # Wait, if we enforce sort, it's TL-TR-BL-BR.
            # To draw a rect, we need TL->TR->BR->BL.
            # 0(TL) -> 1(TR) -> 3(BR) -> 2(BL) based on reorder logic from board_detection?
            # Reorder: 0:TL, 1:TR, 2:BL, 3:BR.
            # So Rect path: 0 -> 1 -> 3 -> 2 -> 0.
            
            # Let's just reorder for drawing to be sure
            reshaped = np.array(self.corners).reshape((4, 1, 2))
            ord_pts = board_detection.reorder(reshaped) # TL, TR, BL, BR
            # path: TL, TR, BR, BL
            draw_cnt = np.array([ ord_pts[0][0], ord_pts[1][0], ord_pts[3][0], ord_pts[2][0] ], np.int32)
            cv2.polylines(img, [draw_cnt], True, (0, 255, 0), 2)

        # 2. Draw Preview Overlay
        if img_preview is not None:
            # Resize preview to be smaller and put in corner
            h, w, c = img_preview.shape
            scale = 0.3
            new_w, new_h = int(w*scale), int(h*scale)
            prev_small = cv2.resize(img_preview, (new_w, new_h))
            
            # Add Border
            prev_small = cv2.copyMakeBorder(prev_small, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            
            # Overlay on bottom right
            y_offset = self.height - new_h - 20
            x_offset = self.width - new_w - 20
            img[y_offset:y_offset+new_h+4, x_offset:x_offset+new_w+4] = prev_small
            
            cv2.putText(img, "Preview (Orientado)", (x_offset, y_offset - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

            # Draw Labels on Preview
            # If Player=White, draw 'a1' at bottom-left
            # If Player=Black, draw 'h8' at bottom-left
            # The draw_chess_grid function ALWAYS draws a1 at bottom-left of the image passed to it.
            # Wait, let's check board_detection.draw_chess_grid.
            # It draws files a-h and ranks 1-8 relative to the image geometry.
            # Files (a-h) on bottom edge.
            # Ranks (1-8) on left edge (8 at top, 1 at bottom).
            
            # This logic matches "Standard/White" view.
            # If we are viewing as Black (Rotated 180), we effectively see:
            # Image Bottom = Rank 8 (physically)
            # Image Left = File H (physically)
            
            # BUT `draw_chess_grid` hardcodes labels.
            # So if we just pass the rotated image to `draw_chess_grid`, it will label it as if it's White view.
            # i.e. It will label the visually bottom-left corner as 'a1'.
            # BUT if I am Black, that corner IS 'h8'.
            # So I need a CUSTOM draw function or modify `draw_chess_grid` to accept 'orientation'.
            
            # Let's locally implement a simple label drawer for preview.
            lbl_img = prev_small
            h_s, w_s = lbl_img.shape[:2]
            
            label_color = (0, 255, 255)
            # Bottom-Left Label
            bl_text = "a1" if self.player_color == "WHITE" else "h8"
            cv2.putText(lbl_img, bl_text, (5, h_s - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
            
            # Top-Right Label
            tr_text = "h8" if self.player_color == "WHITE" else "a1"
            cv2.putText(lbl_img, tr_text, (w_s - 30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
            
            # Side Labels based on choice?
            # Let's trust the corners for now.

    def run(self, cap):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("Iniciando Calibracao Interativa...")
        
        while True:
            success, img = cap.read()
            if not success:
                print("Falha na camera.")
                break
                
            img_display = img.copy()
            
            # Generate Preview
            img_preview = self.get_warped_preview(img)
            
            # Draw UI
            self.draw_ui(img_display, img_preview)
            
            cv2.imshow(self.window_name, img_display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Saindo sem salvar.")
                cv2.destroyWindow(self.window_name)
                return None
            
            elif key == 13: # ENTER
                if len(self.corners) == 4:
                    self.sort_corners() # Final sort to be sure
                    self.save_calibration()
                    print("[calibracao] concluida — jogador: " + self.player_color + " — orientacao aplicada")
                    cv2.destroyWindow(self.window_name)
                    return {
                        "corners": self.corners,
                        "player_color": self.player_color,
                        "orientation_flipped": self.orientation_flipped
                    }
                else:
                    print("Defina 4 cantos antes de confirmar.")
            
            elif key == ord('r'):
                self.corners = []
                self.dragging_idx = -1
                print("Cantos resetados.")
            
            elif key == ord('f'):
                self.orientation_flipped = not self.orientation_flipped
                print(f"Orientacao fisica invertida: {self.orientation_flipped}")
                
            elif key == ord('p'):
                if self.player_color == "WHITE":
                    self.player_color = "BLACK"
                else:
                    self.player_color = "WHITE"
                print(f"Jogador alterado para: {self.player_color}")

        return None
