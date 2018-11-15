import copy
import numpy as np
from collections import defaultdict


class Agent(object):
    '''
    A class that represents an AI opponent. There are two different policies
    this agent acts by: a random policy and a policy that uses simulated
    games to estimate the probability of winning the game if they are
    to take a certain action.
    '''

    def __init__(self, difficulty=1000):
        '''
        Arguments:
                difficulty: the number of simulations to run at each decision.
        '''
        self.difficulty = difficulty

        # Stores the wins, losses, and draws.
        self.wins = 0
        self.losses = 0
        self.draws = 0

        # Stores the value of each action at each state.
        self.grids = []

    def move_selection(self, state):
        '''
        Calculates valid moves in a given state.

        Arguments:
                state: the current state of the board.

        Returns:
                possible_move: the valid moves in this state.
        '''
        possible_moves = np.where(state == 5)
        xs, ys = possible_moves
        possible_moves = np.concatenate(
            (xs.reshape(-1, 1), ys.reshape(-1, 1)), axis=1)

        return possible_moves

    def select_action(self, state, policy='random', board=None):
        '''
        Selects an action based on a policy that is being followed
        by the agent.

        Arguments:
                state: the current state of the board
                policy: the policy being followed
                board: the board object
        '''

        # Calculate the possible moves in this state.
        possible_moves = self.move_selection(state)

        # If the policy is random, choose the moves uniformly.
        if policy == 'random':
            number_of_moves = possible_moves.shape[0]

            index = np.random.choice(number_of_moves)
            move = tuple(possible_moves[index])

            return move

        # If the policy is 'search', perform simulations.
        elif policy == 'search':
            # A grid that holds the potential worth of each action.
            potential_grid = np.zeros((3, 3))

            # A dictionary that wholes the worth of each action.
            action_dict = defaultdict(int)

            # If there is only one possible move left, choose it.
            if possible_moves.shape[0] == 1:
                return possible_moves[0]

            # Else, simulate a number of games and store the results
            for i in range(self.difficulty):
                virtual_board = copy.deepcopy(board)
                reward, action = virtual_board.simulate_game(self)
                action_dict[action] += reward

            # Fill the grid (this is for heatmapping)
            for key, value in action_dict.items():
                x, y = key
                potential_grid[x, y] = value

            self.grids.append(potential_grid)

            # Choose the action that is most likely to win
            # against a random agent.
            action = max(action_dict, key=action_dict.get)

            return action


class TicTacToe(object):
    '''
    A class that represents the TicTacToe game.
    '''

    def __init__(self):
        self.board = np.zeros((3, 3)) + 5
        self.player = 'x'
        self.complete = False
        self.number_of_moves = 0

    def reset(self):
        '''
        Resets the state of the board to the beginning
        '''
        self.player = 'x'
        self.board = np.zeros((3, 3)) + 5
        self.number_of_moves = 0
        self.complete = False

    def move(self, position):
        '''
        Performs a move (places a piece) given the position

        Arguments:
                position: the position of the piece in the grid.

        Returns:
                next_state: the next state of the board
                reward: the reward for moving to this state.
        '''
        i, j = position

        if self.board[i, j] != 5:
            print("Invalid move, try again")
            return None, None

        if self.player == 'o':
            piece = 0
            self.player = 'x'

        elif self.player == 'x':
            piece = 1
            self.player = 'o'

        self.board[i, j] = piece
        self.number_of_moves += 1

        next_state = self.board
        reward = self.rewards()

        return next_state, reward

    def rewards(self):
        '''
        Returns:
                reward: the reward for being in a particular state of the board.
        '''
        for i in range(3):
            # X wins verticals or horizontals
            if sum(self.board[:, i]) == 3 or sum(self.board[i, :]) == 3:
                self.complete = True
                return 1

            # O wins verticals or horizontals
            if sum(self.board[:, i]) == 0 or sum(self.board[i, :]) == 0:
                self.complete = True
                return -1

        # X wins forward diagonal
        if self.board[0, 0] + self.board[1, 1] + self.board[2, 2] == 3:
            self.complete = True
            return 1

        # O wins forward diagonal
        if self.board[0, 0] + self.board[1, 1] + self.board[2, 2] == 0:
            self.complete = True
            return -1

        # X wins backward diagonal
        if self.board[0, 2] + self.board[1, 1] + self.board[2, 0] == 3:
            self.complete = True
            return 1

        # O wins backward diagonal
        if self.board[0, 2] + self.board[1, 1] + self.board[2, 0] == 0:
            self.complete = True
            return -1

        # Either non-complete game or a draw
        return 0

    def play_game(self, agent):
        '''
        Instantiates a game between the AI and a human.

        Arguments:
                agent: the AI opponent object.
        '''
        self.reset()
        print("GAME STARTED: Difficulty Level = %d" % agent.difficulty)
        while self.complete is False and self.number_of_moves != 9:
            action = agent.select_action(
                self.board, policy='search', board=self)
            print("Computer played position: %s" % str(tuple(action)))
            _, reward = self.move(action)

            print('\n')
            print(self)
            print('\n')

            if self.number_of_moves == 9 or self.complete is True:
                break

            valid_move = False
            while not valid_move:
                print("Actions to choose from: [1 - 9]")
                action = input("Enter a move: ")

                if int(action) < 1 or int(action) > 9:
                    print("Invalid action, try again!")
                    continue

                convert_action = {'1': '00', '2': '01', '3': '02', '4': '10',
                                  '5': '11', '6': '12', '7': '20', '8': '21', '9': '22'}

                action = convert_action[action]
                action = tuple([int(a) for a in action])

                print("You played position: %s" % str(action))

                _, reward = self.move(action)

                if reward is None:
                    valid_move = False

                else:
                    valid_move = True

            print(self)

        if reward == 1:
            print("You lose!")

        elif reward == -1:
            print("You win!")

        else:
            print("You draw!")

    def computer_play_game(self, agent):
        '''
        Instantiates a game between the AI and a random
        policy agent.

        Arguments:
                agent: the AI opponent object.

        Returns:
                reward: the reward for the end-game state (-1, +1, 0).
        '''
        self.reset()
        while self.complete is False and self.number_of_moves != 9:
            action = agent.select_action(
                self.board, policy='search', board=self)
            _, reward = self.move(action)

            if self.number_of_moves == 9 or self.complete is True:
                break

            action = agent.select_action(self.board)
            _, reward = self.move(action)

        return reward

    def simulate_game(self, agent):
        '''
        Instantiates a simulated game between two agents following
        a random policy.

        Arguments:
                agent: the AI opponent object

        Returns:
                reward: the reward for the end-game state.
                player_move[0]: the first move taken that finally led to
                                the end-game state.
        '''
        player_move = []

        while self.complete is False and self.number_of_moves != 9:
            if self.player == 'x':
                action = agent.select_action(self.board)
                player_move.append(action)
                _, reward = self.move(action)

            elif self.player == 'o':
                action = agent.select_action(self.board)
                _, reward = self.move(action)

        return reward, player_move[0]

    def __str__(self):
        '''
        Pretty-prints the TicTacToe board.
        '''
        board = str(self.board)
        board = board.replace('[', '')
        board = board.replace(']', '')
        board = ' ' + board
        board = board.replace(
            '\n', '\n---------------------------------------------------\n')

        board = board.replace('5.', '|\t\t')
        board = board.replace('0.', '|\tX\t')
        board = board.replace('1.', '|\tO\t')
        board = board.replace('\t\n', '\t|\n')
        board = board + '|'
        return board
