
def parse_fen_grid(fen):
    """
    Parses the board part of a FEN string into an 8x8 list of lists.
    Returns:
        grid: 8x8 list where grid[row][col] is a piece char or None.
              Row 0 is Rank 8, Col 0 is File A.
    """
    board_part = fen.split(' ')[0]
    rows = board_part.split('/')
    grid = []
    
    for row_str in rows:
        row_data = []
        for char in row_str:
            if char.isdigit():
                # Empty squares
                num_empty = int(char)
                row_data.extend([None] * num_empty)
            else:
                # Piece
                row_data.append(char)
        grid.append(row_data)
        
    return grid

def get_algebraic(row, col):
    """
    Converts 0-indexed row/col to algebraic notation (e.g., 6, 4 -> e2).
    Row 0 is Rank 8, Row 7 is Rank 1.
    Col 0 is File a, Col 7 is File h.
    """
    file_char = chr(ord('a') + col)
    rank_num = 8 - row
    return f"{file_char}{rank_num}"

def detect_move(prev_fen, curr_fen):
    """
    Compares two FEN strings and infers the move.
    
    Rules for valid simple move:
    1. Exactly 2 squares changed.
    2. One square went from Occupied -> Empty (Source).
    3. One square went from Empty -> Occupied (Destination).
    4. Simple check: The piece at Dest must match the piece that left Source (optional, but good for sanity).
       Actually, just checking occ->empty and empty->occ is enough for "something moved here".
       But validating piece consistency helps filter noise.
       
    Returns:
        move_str: e.g. "e2e4" or None if invalid/ambiguous.
    """
    if not prev_fen or not curr_fen:
        return None
        
    grid_prev = parse_fen_grid(prev_fen)
    grid_curr = parse_fen_grid(curr_fen)
    
    diffs = []
    
    for r in range(8):
        for c in range(8):
            p_prev = grid_prev[r][c]
            p_curr = grid_curr[r][c]
            
            if p_prev != p_curr:
                diffs.append(((r, c), p_prev, p_curr))
    
    # Must have exactly 2 differences
    if len(diffs) != 2:
        return None
        
    # Analyze differences
    source = None
    dest = None
    
    # Sort or iterate to find which is source and which is dest
    for (r, c), p_old, p_new in diffs:
        # Source: Was Occupied, Now Empty
        if p_old is not None and p_new is None:
            if source is not None:
                return None # Multiple sources? Invalid for simple move
            source = ((r, c), p_old)
            
        # Destination: Was Empty, Now Occupied
        elif p_old is None and p_new is not None:
            if dest is not None:
                return None # Multiple dests?
            dest = ((r, c), p_new)
            
        else:
            # Case: Occupied -> Occupied (Capture or piece change?)
            # This version does NOT support captures (per user rules: dest was empty)
            return None

    if source and dest:
        (src_pos, src_piece) = source
        (dst_pos, dst_piece) = dest
        
        # Identity check (sanity): The piece appearing at dest should be the one that left source
        # Note: If we had precise piece recognition, this must match. 
        # If recognition is noisy 'P' vs 'p' or 'B' vs 'P', this might fail valid moves.
        # But for robust system, we assume classifier is correct.
        if src_piece != dst_piece:
             # Allowed if we want to be lenient? 
             # For now, strict: piece shouldn't change identity mid-move (unless promotion - excluded).
             return None 
             
        return get_algebraic(*src_pos) + get_algebraic(*dst_pos)
        
    return None
