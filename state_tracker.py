from collections import Counter, deque

class StateTracker:
    def __init__(self, history_length=5):
        self.history_length = history_length
        self.fen_history = deque(maxlen=history_length)
        self.stable_fen = None
        self.last_stable_fen = None
        
    def update(self, current_fen):
        """
        Updates the state with a new FEN candidate.
        Returns:
            stable_fen: The current stable FEN (most common in history).
            changed: Boolean, True if stable_fen changed since last confirmation.
        """
        # We only care about the board position part of the FEN for stability
        truncated_fen = current_fen.split(' ')[0]
        self.fen_history.append(truncated_fen)
        
        # Determine most common FEN in history
        if len(self.fen_history) < self.history_length:
             return None, False

        count = Counter(self.fen_history)
        most_common_fen, frequency = count.most_common(1)[0]
        
        # Define stability threshold (e.g., must be majority)
        if frequency >= (self.history_length // 2) + 1:
            self.stable_fen = most_common_fen
            
            if self.stable_fen != self.last_stable_fen:
                # State has changed
                self.last_stable_fen = self.stable_fen
                # Reconstruct full FEN (using passed turn/extras from current, or default)
                full_stable_fen = current_fen # Or reconstruct carefully if turn matters
                return full_stable_fen, True
            else:
                return current_fen, False # Stable but not changed
        
        return None, False

    def get_current_fen(self):
        return self.stable_fen
