from enum import Enum, auto
from move_detector import detect_move

class DetectionState(Enum):
    IDLE = auto()       # Monitoring for changes (Stable state matches confirmed)
    TRANSITION = auto() # Potential move in progress (Candidate != confirmed)
    CONFIRMED = auto()  # New stable state confirmed (Ready to commit)

class StableMoveDetector:
    def __init__(self, stability_threshold=15):
        """
        Args:
            stability_threshold: Number of consecutive frames a FEN must persist 
                                 to be considered 'confirmed'.
        """
        self.stability_threshold = stability_threshold
        
        # The last confirmed stable state of the board
        self.confirmed_fen = None
        
        # State Machine
        self.state = DetectionState.IDLE
        
        # Candidate tracking
        self.candidate_fen = None
        self.candidate_count = 0
        
    def process(self, current_fen):
        """
        Process a new FEN frame through the state machine.
        
        Returns:
            move (str or None): The detected move if a NEW stable state is confirmed.
        """
        if current_fen is None:
            return None

        # 1. Initialization (First run)
        if self.confirmed_fen is None:
            if current_fen == self.candidate_fen:
                self.candidate_count += 1
            else:
                self.candidate_fen = current_fen
                self.candidate_count = 1
                
            if self.candidate_count >= self.stability_threshold:
                self.confirmed_fen = self.candidate_fen
                self.state = DetectionState.IDLE
                # No move returned on initial stabilization
            return None

        # 2. State Machine Logic
        
        # --- IDLE STATE ---
        if self.state == DetectionState.IDLE:
            if current_fen != self.confirmed_fen:
                # Change detected, start tracking candidate
                self.state = DetectionState.TRANSITION
                self.candidate_fen = current_fen
                self.candidate_count = 1
            else:
                # Still stable, nothing to do
                pass
                
        # --- TRANSITION STATE ---
        elif self.state == DetectionState.TRANSITION:
            if current_fen == self.candidate_fen:
                self.candidate_count += 1
                
                # Check if stability threshold reached
                if self.candidate_count >= self.stability_threshold:
                    self.state = DetectionState.CONFIRMED
            else:
                # Candidate changed mid-transition (instability or new move started)
                # Reset candidate tracking
                if current_fen == self.confirmed_fen:
                    # Reverted to original state (false alarm)
                    self.state = DetectionState.IDLE
                    self.candidate_fen = None
                    self.candidate_count = 0
                else:
                    # Switched to a different candidate
                    self.candidate_fen = current_fen
                    self.candidate_count = 1
                    
        # --- CONFIRMED STATE ---
        # Note: We process this immediately after checking transition to avoid 1-frame delay
        if self.state == DetectionState.CONFIRMED:
            # 1. Detect the move (Compare OLD Confirmed -> NEW Candidate)
            # Ensure we are comparing different states
            move = None
            if self.candidate_fen != self.confirmed_fen:
                move = detect_move(self.confirmed_fen, self.candidate_fen)
            
            # 2. Commit the new state
            self.confirmed_fen = self.candidate_fen
            
            # 3. Reset to IDLE
            self.state = DetectionState.IDLE
            self.candidate_fen = None
            self.candidate_count = 0
            
            # 4. Return the move (only if valid)
            return move
            
        return None

    def get_status(self):
        """Returns debug info string."""
        if self.confirmed_fen is None:
            return "Calibrando..."
        
        if self.state == DetectionState.IDLE:
             return "Aguardando movimento..."
             
        elif self.state == DetectionState.TRANSITION:
            return f"Estabilizando... ({self.candidate_count}/{self.stability_threshold})"
            
        elif self.state == DetectionState.CONFIRMED:
            return "Confirmado!"
            
        return "Unknown"
