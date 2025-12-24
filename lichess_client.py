"""
Lichess Client - API Integration

Uses direct HTTP requests to communicate with Lichess Board API.
Compatible with Python 3.13+.
"""

import os
import json
import threading
from typing import Optional, Generator
import requests
from dotenv import load_dotenv


class LichessClient:
    """
    Client for Lichess Board API using direct HTTP requests.
    
    Usage:
        client = LichessClient()
        if client.connect():
            for event in client.stream_game("game_id"):
                print(event)
    """
    
    BASE_URL = "https://lichess.org"
    
    def __init__(self):
        load_dotenv()
        self.token = os.getenv("LICHESS_TOKEN")
        self.username: Optional[str] = None
        self.current_game_id: Optional[str] = None
        self.my_color: Optional[str] = None
        self._headers = {}
        
    def connect(self) -> bool:
        """Connect to Lichess using token from .env"""
        if not self.token:
            print("[!] LICHESS_TOKEN not found in .env")
            return False
        
        self._headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json"
        }
        
        try:
            # Test connection by getting account info
            response = requests.get(
                f"{self.BASE_URL}/api/account",
                headers=self._headers,
                timeout=10
            )
            
            if response.status_code == 200:
                account = response.json()
                self.username = account.get("username")
                print(f"[Lichess] Connected as: {self.username}")
                return True
            else:
                print(f"[!] Lichess API error: {response.status_code}")
                return False
                
        except requests.RequestException as e:
            print(f"[!] Connection failed: {e}")
            return False
    
    def get_ongoing_games(self) -> list:
        """Get list of ongoing games."""
        try:
            response = requests.get(
                f"{self.BASE_URL}/api/account/playing",
                headers=self._headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("nowPlaying", [])
            return []
            
        except Exception as e:
            print(f"[!] Error getting games: {e}")
            return []
    
    def stream_game(self, game_id: str) -> Generator:
        """
        Stream game events in real-time using NDJSON.
        
        Yields events like:
        - {"type": "gameFull", "white": {...}, "black": {...}, "state": {...}}
        - {"type": "gameState", "moves": "e2e4 e7e5", "status": "started"}
        """
        self.current_game_id = game_id
        
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/x-ndjson"
        }
        
        try:
            response = requests.get(
                f"{self.BASE_URL}/api/board/game/stream/{game_id}",
                headers=headers,
                stream=True,
                timeout=None  # Long-lived connection
            )
            
            if response.status_code != 200:
                print(f"[!] Stream error: {response.status_code}")
                return
            
            for line in response.iter_lines():
                if line:
                    try:
                        event = json.loads(line.decode('utf-8'))
                        
                        # Set my color on gameFull event
                        if event.get("type") == "gameFull":
                            self._set_my_color(event)
                        
                        yield event
                        
                    except json.JSONDecodeError:
                        continue
                        
        except requests.RequestException as e:
            print(f"[!] Stream error: {e}")
    
    def _set_my_color(self, event: dict):
        """Determine which color we're playing."""
        white = event.get("white", {})
        black = event.get("black", {})
        
        white_id = white.get("id", "").lower()
        black_id = black.get("id", "").lower()
        my_id = (self.username or "").lower()
        
        if white_id == my_id:
            self.my_color = "white"
        elif black_id == my_id:
            self.my_color = "black"
        
        print(f"[Lichess] Playing as: {self.my_color}")
    
    def make_move(self, uci_move: str) -> bool:
        """
        Send a move to Lichess.
        
        Args:
            uci_move: Move in UCI format, e.g., "e2e4"
            
        Returns:
            True if move was accepted
        """
        if not self.current_game_id:
            print("[!] No active game")
            return False
        
        try:
            response = requests.post(
                f"{self.BASE_URL}/api/board/game/{self.current_game_id}/move/{uci_move}",
                headers=self._headers,
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"[Lichess] Move sent: {uci_move}")
                return True
            else:
                print(f"[!] Move rejected: {response.status_code} - {response.text}")
                return False
                
        except requests.RequestException as e:
            print(f"[!] Move error: {e}")
            return False
    
    def resign(self) -> bool:
        """Resign the current game."""
        if not self.current_game_id:
            return False
        try:
            response = requests.post(
                f"{self.BASE_URL}/api/board/game/{self.current_game_id}/resign",
                headers=self._headers,
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
    
    def is_my_turn(self, moves_str: str) -> bool:
        """Check if it's my turn based on moves string."""
        if not self.my_color:
            return False
            
        moves = moves_str.split() if moves_str else []
        move_count = len(moves)
        
        if self.my_color == "white":
            return move_count % 2 == 0
        else:
            return move_count % 2 == 1
    
    def get_last_move(self, moves_str: str) -> Optional[str]:
        """Get the last move from moves string."""
        if not moves_str:
            return None
        moves = moves_str.split()
        return moves[-1] if moves else None
    
    def seek_game(self, time_minutes: int = 10, increment: int = 0, rated: bool = False) -> Optional[str]:
        """
        Seek a new game.
        
        Returns game_id if matched, None otherwise.
        """
        try:
            data = {
                "time": time_minutes,
                "increment": increment,
                "rated": rated
            }
            
            response = requests.post(
                f"{self.BASE_URL}/api/board/seek",
                headers={**self._headers, "Accept": "application/x-ndjson"},
                data=data,
                stream=True,
                timeout=30
            )
            
            for line in response.iter_lines():
                if line:
                    event = json.loads(line.decode('utf-8'))
                    if "id" in event:
                        return event["id"]
            
            return None
            
        except Exception as e:
            print(f"[!] Seek error: {e}")
            return None


def test_connection():
    """Test Lichess connection."""
    client = LichessClient()
    if client.connect():
        games = client.get_ongoing_games()
        print(f"[Lichess] Ongoing games: {len(games)}")
        for g in games:
            game_id = g.get("gameId", g.get("id", "?"))
            opponent = g.get("opponent", {}).get("username", "?")
            print(f"  - {game_id}: vs {opponent}")
        return True
    return False


if __name__ == "__main__":
    test_connection()
