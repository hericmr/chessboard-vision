import threading
import time
import chess
from unittest import mock
from lichess_session import LichessSession
from game_session import GameSession
from game_state import GameState
from noise_handler import NoiseState

# Mock LichessClient
class MockLichessClient:
    def __init__(self):
        self.game_id = "test_game"
        self.my_color = "white"

    def connect(self):
        return True
    
    def get_ongoing_games(self):
        return [{"gameId": "test_game", "opponent": {"username": "bot"}}]

    def is_my_turn(self, moves):
        if not moves: return True # White starts
        return len(moves.split()) % 2 == 0

    def make_move(self, uci):
        return True

    def get_last_move(self, moves):
        if not moves: return None
        return moves.split()[-1]

    def stream_game(self, game_id):
        # Mock stream basic behavior
        yield {"type": "gameFull", "state": {"moves": ""}}
        while True:
            time.sleep(1)

def test_race_condition():
    print("Initializing Session...")
    session = LichessSession()
    session.lichess = MockLichessClient()
    
    session.game = GameState()
    session.game.board = chess.Board() # Start pos
    session.game.board.turn = chess.WHITE # White to move
    session.noise = mock.MagicMock()
    session.piece_detector = mock.MagicMock()
    
    # We want to simulate catching a move "e2e4"
    move = chess.Move.from_uci("e2e4")
    
    # Mock make_move to trigger background update
    original_make_move = session.lichess.make_move
    
    def delayed_stream_update():
        print("[BG] Stream update thread started. Waiting for lock or opportunity...")
        # Simulate receiving "e2e4" from Lichess stream
        session._sync_moves("e2e4")
        print("[BG] Stream update finished.")

    def mock_make_move(uci):
        print(f"[Main] make_move({uci}) called. Triggering background stream update...")
        # Launch background thread that tries to update board
        t = threading.Thread(target=delayed_stream_update)
        t.start()
        # Sleep a tiny bit to force the background thread to attempt acquiring lock
        time.sleep(0.2) 
        return True

    session.lichess.make_move = mock_make_move
    
    # Mock _infer_move to return "e2e4" to skip vision logic
    session._infer_move = mock.MagicMock(return_value=move)
    
    print("Starting Race Test...")
    
    # State setup to pass stability checks
    
    # Calculate expected occupancy for start pos
    expected_occ = session.game.get_board_occupancy()
    # Simulate move e2 -> e4
    e2 = (4, 1) # file e=4, rank 2=1
    e4 = (4, 3) # file e=4, rank 4=3
    
    vision_occupied = expected_occ.copy()
    if e2 in vision_occupied: vision_occupied.remove(e2)
    vision_occupied.add(e4)
    
    session.stable_occupancy = vision_occupied.copy()
    session.stable_count = session.STABILITY_REQUIRED
    session.last_move_time = 0
    noise_state = NoiseState.IDLE
    
    try:
        # Call process_stable_move
        session._process_stable_move(vision_occupied, [], noise_state)
        print("Success: _process_stable_move completed without error.")
    except Exception as e:
        print(f"FAILED: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return

    # Verify final state
    print(f"Final Move Stack: {session.game.board.move_stack}")
    if session.game.board.move_stack == [move]:
        print("Success: Board state is correct (1 move).")
    else:
        print(f"FAILED: Board state incorrect.")

    print(f"Final Last Lichess Moves: '{session.last_lichess_moves}'")
    if session.last_lichess_moves == "e2e4":
        print("Success: last_lichess_moves is correct.")
    else:
        print(f"FAILED: last_lichess_moves incorrect.")

if __name__ == "__main__":
    test_race_condition()
