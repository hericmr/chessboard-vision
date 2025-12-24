"""
Noise Handler - State Machine for Move Detection

Handles player hand occlusion and movement stabilization.

States:
- IDLE: No changes detected, waiting
- NOISE_ACTIVE: Too many squares changed (hand detected), blocking
- MOVE_PENDING: 1-2 squares changed, waiting for stability
"""

from enum import Enum, auto


class NoiseState(Enum):
    IDLE = auto()
    NOISE_ACTIVE = auto()
    MOVE_PENDING = auto()


class NoiseHandler:
    """
    State machine for handling noise (player hand) during move detection.
    
    Usage:
        handler = NoiseHandler()
        state, data = handler.process(changed_squares)
        
        if state == NoiseState.NOISE_ACTIVE:
            # Block move processing, show noise indicator
        elif state == NoiseState.MOVE_PENDING:
            if data.get("stable"):
                # Process move with data["squares"]
    """
    
    # Thresholds
    NOISE_THRESHOLD = 3       # More than 3 changed squares = noise
    STABILITY_FRAMES = 12     # Frames to wait before confirming
    COOLDOWN_FRAMES = 5       # Frames with 0 changes to exit noise
    
    def __init__(self):
        self.state = NoiseState.IDLE
        self.pending_squares = set()
        self.stable_count = 0
        self.cooldown_count = 0
        self.last_lifted_square = None  # Track which piece might be lifted
        
    def process(self, changed_squares: set) -> tuple:
        """
        Process a frame's changed squares through the state machine.
        
        Args:
            changed_squares: Set of (file, rank) tuples that changed
            
        Returns:
            (NoiseState, dict): Current state and relevant data
        """
        n_changes = len(changed_squares)
        
        # State transitions
        if self.state == NoiseState.IDLE:
            return self._process_idle(changed_squares, n_changes)
            
        elif self.state == NoiseState.NOISE_ACTIVE:
            return self._process_noise(changed_squares, n_changes)
            
        elif self.state == NoiseState.MOVE_PENDING:
            return self._process_pending(changed_squares, n_changes)
            
        return (self.state, {})
    
    def _process_idle(self, changed_squares, n_changes):
        """IDLE state: waiting for changes."""
        
        if n_changes == 0:
            # Stay idle
            return (NoiseState.IDLE, {"message": "waiting"})
            
        elif n_changes > self.NOISE_THRESHOLD:
            # Too many changes = noise (hand)
            self.state = NoiseState.NOISE_ACTIVE
            self.cooldown_count = 0
            return (NoiseState.NOISE_ACTIVE, {
                "message": "hand_detected",
                "changed_count": n_changes
            })
            
        else:
            # 1-3 changes = potential move
            self.state = NoiseState.MOVE_PENDING
            self.pending_squares = changed_squares.copy()
            self.stable_count = 1
            
            # Track lifted piece (single change)
            if n_changes == 1:
                self.last_lifted_square = list(changed_squares)[0]
            else:
                self.last_lifted_square = None
                
            return (NoiseState.MOVE_PENDING, {
                "message": "detecting",
                "squares": self.pending_squares,
                "lifted": self.last_lifted_square,
                "stable": False,
                "progress": self.stable_count / self.STABILITY_FRAMES
            })
    
    def _process_noise(self, changed_squares, n_changes):
        """NOISE_ACTIVE state: hand detected, waiting for clear."""
        
        if n_changes == 0:
            self.cooldown_count += 1
            if self.cooldown_count >= self.COOLDOWN_FRAMES:
                # Hand removed, return to idle
                self.state = NoiseState.IDLE
                self.cooldown_count = 0
                return (NoiseState.IDLE, {"message": "noise_cleared"})
            else:
                return (NoiseState.NOISE_ACTIVE, {
                    "message": "clearing",
                    "cooldown": self.cooldown_count,
                    "progress": self.cooldown_count / self.COOLDOWN_FRAMES
                })
                
        elif n_changes <= self.NOISE_THRESHOLD:
            # Could be transitioning to move
            self.cooldown_count += 1
            if self.cooldown_count >= self.COOLDOWN_FRAMES:
                # Transition to pending
                self.state = NoiseState.MOVE_PENDING
                self.pending_squares = changed_squares.copy()
                self.stable_count = 1
                return (NoiseState.MOVE_PENDING, {
                    "message": "detecting",
                    "squares": self.pending_squares,
                    "stable": False
                })
            return (NoiseState.NOISE_ACTIVE, {
                "message": "stabilizing",
                "changed_count": n_changes
            })
        else:
            # Still noisy
            self.cooldown_count = 0
            return (NoiseState.NOISE_ACTIVE, {
                "message": "hand_active",
                "changed_count": n_changes
            })
    
    def _process_pending(self, changed_squares, n_changes):
        """MOVE_PENDING state: waiting for stability."""
        
        if n_changes > self.NOISE_THRESHOLD:
            # Noise interrupted
            self.state = NoiseState.NOISE_ACTIVE
            self.pending_squares = set()
            self.stable_count = 0
            self.cooldown_count = 0
            return (NoiseState.NOISE_ACTIVE, {
                "message": "interrupted_by_hand",
                "changed_count": n_changes
            })
            
        elif n_changes == 0:
            # No changes - either move completed or cancelled
            self.stable_count += 1
            if self.stable_count >= self.STABILITY_FRAMES:
                # Stable! Return pending squares for processing
                squares = self.pending_squares.copy()
                self._reset()
                return (NoiseState.IDLE, {
                    "message": "move_ready",
                    "squares": squares,
                    "stable": True
                })
            return (NoiseState.MOVE_PENDING, {
                "message": "stabilizing",
                "squares": self.pending_squares,
                "stable": False,
                "progress": self.stable_count / self.STABILITY_FRAMES
            })
            
        elif changed_squares == self.pending_squares:
            # Same squares still changed - continue counting
            self.stable_count += 1
            if self.stable_count >= self.STABILITY_FRAMES:
                squares = self.pending_squares.copy()
                return (NoiseState.MOVE_PENDING, {
                    "message": "stable_ready",
                    "squares": squares,
                    "stable": True,
                    "progress": 1.0
                })
            return (NoiseState.MOVE_PENDING, {
                "message": "counting",
                "squares": self.pending_squares,
                "lifted": self.last_lifted_square if len(self.pending_squares) == 1 else None,
                "stable": False,
                "progress": self.stable_count / self.STABILITY_FRAMES
            })
            
        else:
            # Different squares - update and reset counter
            self.pending_squares = changed_squares.copy()
            self.stable_count = 1
            
            if n_changes == 1:
                self.last_lifted_square = list(changed_squares)[0]
            else:
                self.last_lifted_square = None
                
            return (NoiseState.MOVE_PENDING, {
                "message": "updated",
                "squares": self.pending_squares,
                "lifted": self.last_lifted_square,
                "stable": False,
                "progress": self.stable_count / self.STABILITY_FRAMES
            })
    
    def _reset(self):
        """Reset to idle state."""
        self.state = NoiseState.IDLE
        self.pending_squares = set()
        self.stable_count = 0
        self.cooldown_count = 0
        self.last_lifted_square = None
    
    def reset(self):
        """Public reset method."""
        self._reset()
    
    def is_blocked(self) -> bool:
        """Returns True if move processing should be blocked."""
        return self.state == NoiseState.NOISE_ACTIVE
    
    def get_state_name(self) -> str:
        """Get human-readable state name."""
        names = {
            NoiseState.IDLE: "IDLE",
            NoiseState.NOISE_ACTIVE: "NOISE",
            NoiseState.MOVE_PENDING: "PENDING"
        }
        return names.get(self.state, "UNKNOWN")
