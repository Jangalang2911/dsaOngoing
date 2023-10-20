from mancala_game import *
from utils import *
from mancala_cmdline import *
from leetcode import Solution
import itertools

if __name__=="__main__":
    #dimension, board = read_initial_board_file('example_board.txt')
    # game = MancalaGameManager(None, "example_board.txt")
    # p1 = AiPlayerInterface("agent_minimax_starter.py", TOP, 10, 0, 0)
    # p2 = AiPlayerInterface("agent_minimax_starter.py", BOTTOM, 10, 0, 0)
    # gui = MancalaCommandLine(game, p1, p2)
    # gui.run()
    s = '1221'
    x = itertools.groupby(s)
    print([(digit, group) for digit, group in x])

