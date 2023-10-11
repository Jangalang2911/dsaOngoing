from mancala_game import *
from utils import *

if __name__=="__main__":
    dimension, board = read_initial_board_file('example_board.txt')
    print(heuristic_basic(board, 0))