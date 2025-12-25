
import threading
import chess
import time
from queue import Queue
from game_session import GameSession
from lichess_client import LichessClient

class LichessSession(GameSession):
    def __init__(self):
        super().__init__()
        self.lichess = LichessClient()
        
        # Lichess Specifics
        self.game_id = None
        self.my_color = None
        self.waiting_for_opponent = False
        self.last_lichess_moves = ""
        self.stop_event = threading.Event()

    def connect_and_setup(self):
        """Connects to Lichess and selects/creates a game."""
        print("[1/2] Conectando ao Lichess...")
        if not self.lichess.connect():
            print("[!] Falha na conexão com Lichess API")
            return False
            
        game_id = self._select_or_create_game()
        if not game_id:
            return False
            
        self.game_id = game_id
        print(f"\n[Lichess] Entrando no jogo: {game_id}")
        
        # Iniciar thread de stream
        thread = threading.Thread(target=self._stream_task, daemon=True)
        thread.start()
        
        # Aguardar sync inicial
        time.sleep(0.5)
        print(f"\n=== JOGO NO LICHESS INICIADO ===")
        return True

    def on_move_detected(self, move):
        """Override: Enviar movimento detectado para o Lichess"""
        if self.waiting_for_opponent:
            print("[!] Não é sua vez! Movimento ignorado.")
            return False
            
        uci = move.uci()
        print(f"[Core] Tentando enviar {uci}...")
        
        if self.lichess.make_move(uci):
            print(f"    [Lichess] Enviado com sucesso ✓")
            self.waiting_for_opponent = True
            
            # Atualiza string local (Lock já garantido pelo caller em GameSession)
            if self.last_lichess_moves:
                self.last_lichess_moves += f" {uci}"
            else:
                self.last_lichess_moves = uci
            return True
        else:
            print(f"    [Lichess] [!] Rejeitado pela API")
            return False

    def _stream_task(self):
        """Ouve eventos do jogo em background"""
        for event in self.lichess.stream_game(self.game_id):
            if self.stop_event.is_set():
                break
                
            event_type = event.get("type")
            
            if event_type == "gameFull":
                self.my_color = self.lichess.my_color
                state = event.get("state", {})
                self._sync_moves(state.get("moves", ""))
                
            elif event_type == "gameState":
                status = event.get("status")
                if status != "started":
                    print(f"\n[Lichess] Jogo terminado: {status}")
                    self.stop_event.set()
                else:
                    moves = event.get("moves", "")
                    self._sync_moves(moves)

    def _sync_moves(self, moves_str):
        if moves_str == self.last_lichess_moves:
            return
            
        with self.board_lock:
            # Re-check inside lock for race condition
            if moves_str == self.last_lichess_moves:
                return

            # Replay board
            self.game.reset()
            if moves_str:
                for uci in moves_str.split():
                    try:
                        self.game.board.push_uci(uci)
                    except:
                        pass
            
            self.last_lichess_moves = moves_str
            
            # Check turn
            is_my_turn = self.lichess.is_my_turn(moves_str)
            self.waiting_for_opponent = not is_my_turn
            
            if not is_my_turn:
                 # Checar último movimento (foi do oponente)
                 last_move = self.lichess.get_last_move(moves_str)
                 if last_move:
                     print(f"\n[Oponente] Jogou: {last_move}")
    
    def on_exit(self):
        self.stop_event.set()

    # --- CLI Utils ---
    def _select_or_create_game(self):
        games = self.lichess.get_ongoing_games()
        if games:
            print("\n=== JOGOS EM ANDAMENTO ===")
            for i, g in enumerate(games):
                gid = g.get("gameId", g.get("id"))
                opp = g.get("opponent", {}).get("username", "?")
                print(f"  [{i+1}] {gid}: vs {opp}")
            print("\n  [0] Criar novo jogo")
            try:
                c = input("Escolha: ")
                if c == "0": return self._wait_for_challenge()
                idx = int(c)-1
                if 0 <= idx < len(games):
                    return games[idx].get("gameId", games[idx].get("id"))
            except: pass
            
        return self._wait_for_challenge()

    def _wait_for_challenge(self):
        print("\n=== AGUARDANDO JOGO ===")
        print("Crie um jogo no Lichess agora...")
        try:
            while not self.stop_event.is_set():
                time.sleep(2)
                games = self.lichess.get_ongoing_games()
                if games:
                    return games[0].get("gameId", games[0].get("id"))
                print(".", end="", flush=True)
        except KeyboardInterrupt:
            return None
