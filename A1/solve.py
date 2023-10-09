############################################################
## CSC 384, Intro to AI, University of Toronto.
## Assignment 1 Starter Code
## v1.1
##
## Changes: 
## v1.1: removed the hfn paramete from dfs. Updated solve_puzzle() accordingly.
############################################################

from typing import List
import heapq
from heapq import heappush, heappop
import time
import argparse
import math # for infinity

from board import *

def is_goal(state):
    """
    Returns True if the state is the goal state and False otherwise.

    :param state: the current state.
    :type state: State
    :return: True or False
    :rtype: bool
    """

    #Approach 2
    box_set = set(state.board.boxes)
    storage_set = set(state.board.storage)
    if box_set==storage_set:
        return True
    return False

    #raise NotImplementedError



#########################################################################
#Path Functions

def get_path(state):
    """
    Return a list of states containing the nodes on the path 
    from the initial state to the given state in order.

    :param state: The current state.
    :type state: State
    :return: The path.
    :rtype: List[State]
    """
    return recursive_trace(state, [])



def recursive_trace(state, path: list):
    if state.parent==None:
        path.append(state)
        return path
    
    else:
        path = recursive_trace(state.parent, path)
        path.append(state)
        return path


#########################################################################
#Move Functions


def get_successors(state):
    """
    Return a list containing the successor states of the given state.
    The states in the list may be in any arbitrary order.

    :param state: The current state.
    :type state: State
    :return: The list of successor states.
    :rtype: List[State]
    """

    board = state.board
    successor_states=[]

    for robot in board.robots:
        moves = identify_moves(robot, state)
        for move in moves:
            successor_states.append(fetch_successor(robot, move, state))

    return successor_states


def neighbourhood(position: tuple[int, int], board):
    """
    Returns a list containing the neighbourhood of the
    entity at the given position

    Args:
        position (tuple[int, int]): Position of the entity
        board (_type_): Current board

    Returns:
        list[tuple[int, int]]: List of the neighbouring coordinates 
        for an entity
    """
    neighbourhood = set()
    neighbourhood.add((position[0]+1, position[1]))
    neighbourhood.add((position[0]-1, position[1]))
    neighbourhood.add((position[0], position[1]+1))
    neighbourhood.add((position[0], position[1]-1))
        
    return neighbourhood


def identify_moves(robot: tuple[int, int], state):
    """
    Returns a list of available moves for the given robot 
    by analyzing its neighbourhood.

    Args:
        robot (tuple[int]): Position of the robot
        state (_type_): Current state

    Returns:
        list[tuple[int, int]]: List of the possible moves 
    """
    robot_nbd = neighbourhood(robot, state.board)

    #accounting for obstacles in robot's neighbourhood
    robot_nbd -= set(state.board.obstacles)


    #robot->box->wall: movement not possible
    robot_adjacent = robot_nbd.intersection(set(state.board.boxes))
    for box in robot_adjacent:
        #box_nbd: set containing obstacles adjacent to box
        box_nbd=neighbourhood(box, state.board).intersection(
            set(state.board.obstacles))

        #if robot->box->wall, then direction(r->b)=direction(b->w)
        direction=(robot[0]-box[0], robot[1]-box[1])

        invalid_check=[(box[0]-obs[0], box[1]-obs[1])==direction 
                          for obs in box_nbd]
        
        if True in invalid_check:
            robot_nbd -= set([box])

    return list(robot_nbd)


def fetch_successor(robot: tuple[int, int], move: tuple[int, int], state):
    """
    Returns the successor for the provided state, given the position 
    of the robot and the next move.

    Args:
        robot (tuple[int, int]): Position of robot
        move (tuple[int, int]): Position resulting from move
        state (State): Current state
        hfn: Heuristic function

    Returns:
        State: The state resulting from move
    """
    new_board = Board(state.board.name, state.board.width, state.board.height,
                      state.board.robots.copy(), state.board.boxes.copy(),
                      state.board.storage.copy(), state.board.obstacles.copy())
    
    successor = State(board=new_board, hfn=state.hfn, f=0, depth=state.depth+1, 
                      parent=state)
    successor.board.boxes = set(successor.board.boxes)    
    successor.board.robots = set(successor.board.robots)
    successor.board.storage = set(successor.board.storage)
    successor.board.obstacles = set(successor.board.obstacles)

    #move results in pushing a box
    if move in successor.board.boxes:
        direction = (robot[0]-move[0], robot[1]-move[1])
        box_update = (move[0]-direction[0], move[1]-direction[1])

        successor.board.boxes.remove(move) #move and box refer to the same position
        successor.board.boxes.add(box_update)

    successor.board.robots.remove(robot)
    successor.board.robots.add(move)


    successor.board.boxes = list(successor.board.boxes)    
    successor.board.robots = list(successor.board.robots)
    successor.board.storage = list(successor.board.storage)
    successor.board.obstacles = list(successor.board.obstacles)

    
    f_val = successor.depth + state.hfn(new_board)
    successor.f = f_val

    return successor


#########################################################################
# Search Algorithms 
    


def dfs(init_board):
    """
    Run the DFS algorithm given an initial board.

    If the function finds a goal state, it returns a list of states representing
    the path from the initial state to the goal state in order and the cost of
    the solution found.
    Otherwise, it returns am empty list and -1.

    :param init_board: The initial board.
    :type init_board: Board
    :return: (the path to goal state, solution cost)
    :rtype: List[State], int
    """
    init_state = State(board=init_board, hfn=lambda x: 0, f=0, depth=0,
                  parent=None)
    frontier = [init_state]
    explored=set()
    
    while not len(frontier)==0:
        state = frontier.pop()
        
        if state.board in explored: 
            continue

        elif state.board not in explored:
            explored.add(state.board)

            if is_goal(state):
                return get_path(state), state.depth
        
            else:
                frontier.extend(get_successors(state))
        

    return [], -1
    


def a_star(init_board, hfn):
    """
    Run the A_star search algorithm given an initial board and a heuristic function.

    If the function finds a goal state, it returns a list of states representing
    the path from the initial state to the goal state in order and the cost of
    the solution found.
    Otherwise, it returns am empty list and -1.

    :param init_board: The initial starting board.
    :type init_board: Board
    :param hfn: The heuristic function.
    :type hfn: Heuristic (a function that consumes a Board and produces a numeric heuristic value)
    :return: (the path to goal state, solution cost)
    :rtype: List[State], int
    """
    f_val = hfn(init_board)
    init_state = State(board=init_board, hfn=hfn, f=f_val, depth=0,
                  parent=None)
    frontier = [init_state]
    explored=set()
    
    while not len(frontier)==0:
        state = heappop(frontier)
        

        if state.board in explored: 
            continue

        elif state.board not in explored:
            explored.add(state.board)

            if is_goal(state):
                return get_path(state), state.depth
        
            else:
                successors = get_successors(state)
                for successor in successors:
                    heappush(frontier, successor)
    return [], -1


def heuristic_basic(board):
    """
    Returns the heuristic value for the given board
    based on the Manhattan Distance Heuristic function.

    Returns the sum of the Manhattan distances between each box 
    and its closest storage point.

    :param board: The current board.
    :type board: Board
    :return: The heuristic value.
    :rtype: int
    """
    heuristic_val = 0
    boxes = board.boxes
    storage = board.storage
    occupied_storage = set(storage).intersection(set(boxes))
    pairs = {}
    
    for each in occupied_storage:
        pairs[each] = each
    #dictionary mapping a box to a storage point and the distance between them

    for box in boxes:
         if box in pairs: continue

         elif box not in pairs:
             candidates = [store for store in storage if store not in pairs.values()]
             distances = [abs(box[0]-store[0])+abs(box[1]-store[1]) for store in candidates]
             min_dist = min(distances)
             min_index = distances.index(min_dist)
             assigned_store = candidates[min_index]
             pairs[box] = assigned_store

    for box in pairs:
        distance = abs(box[0]-pairs[box][0]) + abs(box[1]-pairs[box][1])
        heuristic_val += distance

    return heuristic_val


def heuristic_advanced(board):
    """
    Returns heuristic value for the given board based on the number of deadlocks.
    A deadlock is a position which makes the board unsolvable.
    Different types of deadlocks are penalized differently.

    Args:
        board (_type_): Current board
    """
    heuristic = 0
    boxes = board.boxes

    corners = [is_corner(box, board) for box in boxes]
    heuristic = heuristic_basic(board)
    if any(corners):
        heuristic+= math.inf

    return heuristic



def is_corner(box: tuple[int, int], board):
    """
    Returns True if the box is located
        1. at a corner
        2. along a wall with no available storage
           (eventually leading to a corner)
    A corner may consist of only obstacles, or both boxes and obstacles

    Args:
        box (tuple[int, int]): Position of box
    """
    box_nbd = neighbourhood(box, board)
    storage = board.storage
    surrounding =  box_nbd.intersection(set(board.obstacles))
    surrounding = surrounding.intersection(set(board.boxes))

    #at corner
    if len(surrounding) > 2 and box not in storage:
        return True
    #at corner
    if len(surrounding) ==2 and box not in storage:
        wall0 = surrounding[0]
        wall1 = surrounding[1]
        direction0 = (wall0[0]-box[0], wall0[1]-box[1])
        direction1 = (box[0]-wall1[0], box[1]-wall1[1])
        if direction0!=direction1:
            return True
        

    #along a wall, leading to a corner
    if len(surrounding)==1 and surrounding[0] in board.obstacles and box not in storage:
        direction = (box[0]-surrounding [0], box[1]-surrounding [1])
        if direction in [(1, 0), (-1, 0)]:
            check_storage =[True for store in board.storage if box[0]==store[0]]
            if not any(check_storage):
                return True
        elif direction in [(0, 1), (0,-1)]:
            check_storage = [True for store in board.storage if box[1]==store[1]]
            if not any(check_storage):
                return True

    return False

def solve_puzzle(board: Board, algorithm: str, hfn):
    """
    Solve the given puzzle using the given type of algorithm.

    :param algorithm: the search algorithm
    :type algorithm: str
    :param hfn: The heuristic function
    :type hfn: Optional[Heuristic]

    :return: the path from the initial state to the goal state
    :rtype: List[State]
    """

    print("Initial board")
    board.display()

    time_start = time.time()

    if algorithm == 'a_star':
        print("Executing A* search")
        path, step = a_star(board, hfn)
    elif algorithm == 'dfs':
        print("Executing DFS")
        path, step = dfs(board)
    else:
        raise NotImplementedError

    time_end = time.time()
    time_elapsed = time_end - time_start

    if not path:

        print('No solution for this puzzle')
        return []

    else:

        print('Goal state found: ')
        path[-1].board.display()

        print('Solution is: ')

        counter = 0
        while counter < len(path):
            print(counter + 1)
            path[counter].board.display()
            print()
            counter += 1

        print('Solution cost: {}'.format(step))
        print('Time taken: {:.2f}s'.format(time_elapsed))

        return path


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The file that contains the puzzle."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The file that contains the solution to the puzzle."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=['a_star', 'dfs'],
        help="The searching algorithm."
    )
    parser.add_argument(
        "--heuristic",
        type=str,
        required=False,
        default=None,
        choices=['zero', 'basic', 'advanced'],
        help="The heuristic used for any heuristic search."
    )
    args = parser.parse_args()

    # set the heuristic function
    heuristic = heuristic_zero
    if args.heuristic == 'basic':
        heuristic = heuristic_basic
    elif args.heuristic == 'advanced':
        heuristic = heuristic_advanced

    # read the boards from the file
    board = read_from_file(args.inputfile)

    # solve the puzzles
    path = solve_puzzle(board, args.algorithm, heuristic)

    # save solution in output file
    outputfile = open(args.outputfile, "w")
    counter = 1
    for state in path:
        print(counter, file=outputfile)
        print(state.board, file=outputfile)
        counter += 1
    outputfile.close()