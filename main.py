import argparse, sys
from tictactoe import TicTacToe, Agent


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--difficulty', help='Set the difficulty!')
    args = parser.parse_args()

    difficulty = int(args.difficulty)
    # Instantiates the agent.
    agent = Agent(difficulty=difficulty)

    # Instantiates the board.
    board = TicTacToe()

    # Plays the game.
    board.play_game(agent)
