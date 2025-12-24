"""
Tests for LichessClient
"""

import unittest
from unittest.mock import patch, MagicMock
from lichess_client import LichessClient


class TestLichessClient(unittest.TestCase):
    def setUp(self):
        self.client = LichessClient()
        self.client.token = "test_token"
        self.client._headers = {"Authorization": "Bearer test_token"}
        self.client.username = "TestUser"
    
    def test_is_my_turn_white(self):
        self.client.my_color = "white"
        
        # Empty = white's turn (move 0)
        self.assertTrue(self.client.is_my_turn(""))
        
        # After 1 move = black's turn
        self.assertFalse(self.client.is_my_turn("e2e4"))
        
        # After 2 moves = white's turn
        self.assertTrue(self.client.is_my_turn("e2e4 e7e5"))
    
    def test_is_my_turn_black(self):
        self.client.my_color = "black"
        
        # Empty = white's turn
        self.assertFalse(self.client.is_my_turn(""))
        
        # After 1 move = black's turn
        self.assertTrue(self.client.is_my_turn("e2e4"))
        
        # After 2 moves = white's turn
        self.assertFalse(self.client.is_my_turn("e2e4 e7e5"))
    
    def test_get_last_move(self):
        self.assertIsNone(self.client.get_last_move(""))
        self.assertEqual(self.client.get_last_move("e2e4"), "e2e4")
        self.assertEqual(self.client.get_last_move("e2e4 e7e5 g1f3"), "g1f3")
    
    @patch('lichess_client.requests.get')
    def test_connect_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"username": "TestUser"}
        mock_get.return_value = mock_response
        
        result = self.client.connect()
        
        self.assertTrue(result)
        self.assertEqual(self.client.username, "TestUser")
    
    @patch('lichess_client.requests.get')
    def test_connect_failure(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response
        
        result = self.client.connect()
        
        self.assertFalse(result)
    
    @patch('lichess_client.requests.post')
    def test_make_move_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        self.client.current_game_id = "game123"
        result = self.client.make_move("e2e4")
        
        self.assertTrue(result)
    
    @patch('lichess_client.requests.post')
    def test_make_move_rejected(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Illegal move"
        mock_post.return_value = mock_response
        
        self.client.current_game_id = "game123"
        result = self.client.make_move("e2e5")  # Illegal
        
        self.assertFalse(result)
    
    def test_make_move_no_game(self):
        self.client.current_game_id = None
        result = self.client.make_move("e2e4")
        
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
