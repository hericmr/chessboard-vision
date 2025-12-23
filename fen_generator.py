
COLUMNS = "abcdefgh"
ROWS = "12345678"

PIECE_TO_FEN = {
    'white-pawn': 'P', 'white-knight': 'N', 'white-bishop': 'B',
    'white-rook': 'R', 'white-queen': 'Q', 'white-king': 'K',
    'black-pawn': 'p', 'black-knight': 'n', 'black-bishop': 'b',
    'black-rook': 'r', 'black-queen': 'q', 'black-king': 'k'
}

def get_chess_square(x, y, board_size):
    """
    Convert pixel coordinates (x, y) in the warped image to chess notation.
    Returns (square notation, grid indices).
    """
    square_size = board_size // 8
    grid_x = x // square_size
    grid_y = y // square_size
    
    # In standard chess matrix:
    # grid_x: 0(a) -> 7(h)
    # grid_y: 0(top, rank 8) -> 7(bottom, rank 1)
    
    if not (0 <= grid_x < 8 and 0 <= grid_y < 8):
        return "fora dos limites", (-1, -1)
    
    col = COLUMNS[grid_x]
    row = ROWS[7 - grid_y]
    return f"{col}{row}", (grid_x, grid_y)

def map_detections_to_board(detections, board_size):
    """
    Maps detections to an 8x8 grid.
    If multiple pieces map to the same square, keeps the one with highest confidence.
    Returns a dictionary {(grid_x, grid_y): piece_fen_char}
    """
    board_map = {}
    
    for det in detections:
        cx, cy = det['center']
        _, (grid_x, grid_y) = get_chess_square(cx, cy, board_size)
        
        if grid_x == -1 or grid_y == -1:
            continue
            
        piece_fen = PIECE_TO_FEN.get(det['class'], '?')
        confidence = det['conf']
        
        # Conflict resolution: Keep highest confidence
        if (grid_x, grid_y) in board_map:
            if confidence > board_map[(grid_x, grid_y)]['conf']:
                board_map[(grid_x, grid_y)] = {'fen': piece_fen, 'conf': confidence, 'class': det['class']}
        else:
            board_map[(grid_x, grid_y)] = {'fen': piece_fen, 'conf': confidence, 'class': det['class']}
            
    return board_map

def generate_fen(board_map, current_turn='w'):
    """
    Generates a FEN string from the board map.
    """
    # Build 8x8 matrix (row 0 = rank 8)
    board = [['' for _ in range(8)] for _ in range(8)]
    
    for (grid_x, grid_y), data in board_map.items():
        board[grid_y][grid_x] = data['fen']
        
    fen_rows = []
    for row in board:
        empty_count = 0
        row_fen = ''
        for cell in row:
            if cell == '':
                empty_count += 1
            else:
                if empty_count > 0:
                    row_fen += str(empty_count)
                    empty_count = 0
                row_fen += cell
        if empty_count > 0:
            row_fen += str(empty_count)
        fen_rows.append(row_fen)
        
    position = '/'.join(fen_rows)
    
    # Simplified castling/ep/moves for now as required
    # Just basic FEN structure
    return f"{position} {current_turn} - - 0 1"
