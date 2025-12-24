#!/usr/bin/env python3
"""
Chess Vision - Simple Move Detection

Minimal version focused on detecting moves using change detection.
"""

import cv2
import numpy as np
import chess
import time

import board_detection
from grid_extractor import GridExtractor
from calibration_module import CalibrationModule
from change_detector import ChangeDetector
from game_state import GameState
from noise_handler import NoiseHandler, NoiseState

# Configuration
CAMERA_ID = 0
WIDTH, HEIGHT = 1280, 720
SKIP_FRAMES = 2  # Process every Nth frame (1=all, 2=50%, 3=33%)


def main():
    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)
    
    # Phase 1: Board Calibration
    print("=== CALIBRAÇÃO DO TABULEIRO ===")
    calib = CalibrationModule()
    config = calib.run(cap)
    
    if config is None:
        print("Calibração cancelada.")
        cap.release()
        cv2.destroyAllWindows()
        return
    
    corners = config["corners"]
    player_color = config["player_color"]
    orientation_flipped = config.get("orientation_flipped", False)
    
    board_corners = np.array(corners).reshape((4, 1, 2))
    points_ordered = board_detection.reorder(board_corners)
    
    # Phase 2: Initialize
    grid = GridExtractor()
    game = GameState()
    detector = ChangeDetector()  # Loads sensitivity from file
    noise = NoiseHandler()  # State machine for noise handling
    
    print(f"\n=== JOGO INICIADO ===")
    print(f"Jogador: {player_color}")
    print("Pressione 'q' para sair, 'c' para recalibrar\n")
    
    # Capture initial reference
    for _ in range(10):
        cap.read()
    
    success, img = cap.read()
    if success:
        warped, _, _ = board_detection.warp_image(img, points_ordered)
        if orientation_flipped:
            warped = cv2.rotate(warped, cv2.ROTATE_180)
        squares = grid.split_board(warped)
        detector.calibrate(squares)
    
    # State tracking (now handled by NoiseHandler)
    last_move_squares = set()
    
    # Performance tracking
    frame_count = 0
    fps_start = time.time()
    fps_display = 0
    
    while True:
        success, img = cap.read()
        if not success:
            break
        
        # FPS counter
        frame_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()
        
        # Skip frames for performance (still capture for camera buffer)
        if SKIP_FRAMES > 1 and frame_count % SKIP_FRAMES != 0:
            cv2.imshow("Camera", img)  # Keep showing camera
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        # Warp board
        warped, _, board_size = board_detection.warp_image(img, points_ordered)
        if orientation_flipped:
            warped = cv2.rotate(warped, cv2.ROTATE_180)
        
        squares = grid.split_board(warped)
        sq_size = board_size // 8
        
        # Detect changes
        changes = detector.detect_changes(squares)
        current_changes = set(changes.keys())
        
        # Process through noise handler state machine
        noise_state, noise_data = noise.process(current_changes)
        
        # Get lifted piece info from noise handler
        lifted_piece_square = noise_data.get("lifted")
        legal_destinations = []
        
        if lifted_piece_square:
            f, r = lifted_piece_square
            sq_idx = chess.square(f, r)
            piece = game.board.piece_at(sq_idx)
            
            if piece and piece.color == game.board.turn:
                for move in game.board.legal_moves:
                    if move.from_square == sq_idx:
                        dest_f = chess.square_file(move.to_square)
                        dest_r = chess.square_rank(move.to_square)
                        legal_destinations.append((dest_f, dest_r))
            else:
                lifted_piece_square = None
        
        # Draw visualization
        vis = warped.copy()
        
        # Draw grid lines
        for i in range(9):
            cv2.line(vis, (i * sq_size, 0), (i * sq_size, board_size), (50, 50, 50), 1)
            cv2.line(vis, (0, i * sq_size), (board_size, i * sq_size), (50, 50, 50), 1)
        
        # NOISE overlay - red tint when hand detected
        if noise_state == NoiseState.NOISE_ACTIVE:
            overlay = vis.copy()
            overlay[:] = (0, 0, 80)  # Dark red tint
            cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
            cv2.putText(vis, "MAO DETECTADA", (board_size//2 - 120, board_size//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        # Draw legal destinations
        elif legal_destinations:
            overlay = vis.copy()
            for dest in legal_destinations:
                df, dr = dest
                dc, drow = df, 7 - dr
                cv2.rectangle(overlay,
                    (dc * sq_size, drow * sq_size),
                    ((dc + 1) * sq_size, (drow + 1) * sq_size),
                    (255, 150, 0), -1)
            cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
        
        # Draw board squares and pieces
        pending_squares = noise_data.get("squares", set())
        for f in range(8):
            for r in range(8):
                pos = (f, r)
                col = f
                row = 7 - r
                x = col * sq_size + sq_size // 2
                y = row * sq_size + sq_size // 2
                
                sq_idx = chess.square(f, r)
                piece = game.board.piece_at(sq_idx)
                
                # Highlight squares
                if pos == lifted_piece_square:
                    cv2.rectangle(vis,
                        (col * sq_size + 3, row * sq_size + 3),
                        ((col + 1) * sq_size - 3, (row + 1) * sq_size - 3),
                        (255, 0, 0), 4)  # Blue = piece lifted
                elif pos in pending_squares:
                    cv2.rectangle(vis,
                        (col * sq_size + 3, row * sq_size + 3),
                        ((col + 1) * sq_size - 3, (row + 1) * sq_size - 3),
                        (0, 255, 255), 3)  # Yellow = changed
                
                # Draw piece symbol
                if piece:
                    sym = piece.symbol()
                    color = (255, 255, 255) if piece.color == chess.WHITE else (0, 0, 0)
                    bg = (0, 0, 0) if piece.color == chess.WHITE else (255, 255, 255)
                    cv2.putText(vis, sym, (x - 15, y + 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, bg, 4)
                    cv2.putText(vis, sym, (x - 15, y + 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        
        # Process move when stable
        if noise_data.get("stable") and noise_data.get("squares"):
            move_squares = noise_data["squares"]
            
            # Build vision occupancy grid for GameState
            # Start with current game state occupancy
            vision_occupied = game.get_board_occupancy().copy()
            
            # For each changed square, toggle its occupancy
            # Changed from occupied → now empty, or empty → now occupied
            for pos in move_squares:
                if pos in vision_occupied:
                    vision_occupied.remove(pos)  # Was occupied, now empty
                else:
                    vision_occupied.add(pos)     # Was empty, now occupied
            
            # Use GameState to resolve the move
            move, status = game.process_occupancy_change(vision_occupied)
            
            if move:
                print(f">>> MOVIMENTO: {move.uci()} ({status})")
                
                if game.board.is_check():
                    print("    XEQUE!")
                if game.board.is_game_over():
                    print(f"    FIM DE JOGO: {game.board.result()}")
            else:
                # Fallback: try simple 2-square detection for edge cases
                if len(move_squares) == 2:
                    sq_list = list(move_squares)
                    pos1, pos2 = sq_list[0], sq_list[1]
                    f1, r1 = pos1
                    f2, r2 = pos2
                    sq1 = chess.square(f1, r1)
                    sq2 = chess.square(f2, r2)
                    
                    piece1 = game.board.piece_at(sq1)
                    piece2 = game.board.piece_at(sq2)
                    
                    if piece1 and not piece2:
                        pos_from, pos_to = pos1, pos2
                    elif piece2 and not piece1:
                        pos_from, pos_to = pos2, pos1
                    elif piece1 and piece2:
                        if piece1.color == game.board.turn:
                            pos_from, pos_to = pos1, pos2
                        else:
                            pos_from, pos_to = pos2, pos1
                    else:
                        pos_from, pos_to = None, None
                    
                    if pos_from and pos_to:
                        f_from, r_from = pos_from
                        f_to, r_to = pos_to
                        move_uci = f"{chr(97+f_from)}{r_from+1}{chr(97+f_to)}{r_to+1}"
                        
                        try:
                            move = chess.Move.from_uci(move_uci)
                            if move in game.board.legal_moves:
                                game.board.push(move)
                                print(f">>> MOVIMENTO: {move_uci}")
                            else:
                                move_uci_rev = f"{chr(97+f_to)}{r_to+1}{chr(97+f_from)}{r_from+1}"
                                move_rev = chess.Move.from_uci(move_uci_rev)
                                if move_rev in game.board.legal_moves:
                                    game.board.push(move_rev)
                                    print(f">>> MOVIMENTO: {move_uci_rev}")
                                else:
                                    print(f"[!] Movimento ilegal: {move_uci}")
                        except Exception as e:
                            print(f"[!] Erro: {e}")
                else:
                    print(f"[!] {status}: {len(move_squares)} casas mudaram")
            
            # Update reference and reset
            detector.update_all_references(squares)
            noise.reset()
        
        # Status display
        state_name = noise.get_state_name()
        progress = noise_data.get("progress", 0)
        
        if noise_state == NoiseState.IDLE:
            status = "AGUARDANDO"
            status_color = (0, 255, 0)
        elif noise_state == NoiseState.NOISE_ACTIVE:
            status = "MAO DETECTADA"
            status_color = (0, 0, 255)
        else:
            status = f"DETECTANDO... {int(progress*100)}%"
            status_color = (0, 255, 255)
        
        cv2.putText(vis, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(vis, f"Turno: {'Brancas' if game.board.turn else 'Pretas'}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis, f"FPS: {fps_display:.1f} | {state_name}", (board_size - 180, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Show
        cv2.imshow("Tabuleiro", vis)
        cv2.imshow("Camera", img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Recapture reference
            detector.calibrate(squares)
            print("[RECALIBRADO]")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()