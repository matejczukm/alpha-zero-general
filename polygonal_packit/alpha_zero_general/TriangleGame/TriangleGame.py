from __future__ import print_function

from ..Game import Game

from ...game_core.triangular_mode.game_logic import *


class TriangleGame(Game):
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """

    def __init__(self, board_side):
        self.board_side = board_side
        self.board_width = board_side * 2 - 1

        number_of_fields = self.board_side ** 2

        # checking the sum of arithmetic series by calculating delta
        self.max_k = int(np.floor((np.sqrt(8*number_of_fields-7)+1)/2))
        # print("max_k: ", self.max_k)

        # placements of shapes of size i begin at partition_indices[i-1] and end at partition_indices[i]-1
        self.partition_indices = np.zeros(self.max_k + 1)

        action_space = []
        board = get_board(self.board_side)
        for k in range(1, self.max_k + 1):
            new_placements = get_possible_placements(board, k, as_list=True)
            action_space.extend(new_placements)
            self.partition_indices[k] = self.partition_indices[k - 1] + len(new_placements)

        self.action_space = np.array(action_space)
        self.partition_indices = self.partition_indices.astype(np.int64)

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        board = get_board(self.board_side)

        return board

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return self.board_side, self.board_width

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return np.shape(self.action_space)[0]

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """

        return board + self.action_space[action], -player

    def getValidMoves(self, board, player, turn):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        if turn > self.max_k:
            return np.zeros(len(self.action_space))

        left_zeros = np.zeros(self.partition_indices[turn - 1])
        if turn < self.max_k:
            action_space_slice = self.action_space[self.partition_indices[turn - 1]:self.partition_indices[turn + 1]]
            valids = validate_placements(action_space_slice, board)
            right_zeros = np.zeros(self.partition_indices[-1] - self.partition_indices[turn + 1])
            return np.hstack((left_zeros, valids, right_zeros))

        action_space_slice = self.action_space[self.partition_indices[turn - 1]:self.partition_indices[turn]]
        valids = validate_placements(action_space_slice, board)

        return np.hstack((left_zeros, valids))

    def getGameEnded(self, board, player, turn):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        if np.sum(self.getValidMoves(board, player, turn)) == 0:
            return -1
        return 0

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        return board

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        pass

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return np.array2string(board)

    @staticmethod
    def display(board):
        print(np.array2string(board))
