import logging

from tqdm import tqdm

log = logging.getLogger(__name__)
import numpy as np

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        board_history = np.copy(board)
        it = 0
        for player in players[0], players[2]:
            if hasattr(player, "startGame"):
                player.startGame()
        
        is_game_over = self.game.getGameEnded(board, curPlayer, 1)
        
        while  is_game_over == 0:
            it += 1

            if verbose:
                assert self.game.display
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.game.display(board)

            action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer), it)
            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1, it)
  

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0

            # Notifying the opponent for the move
            opponent = players[-curPlayer + 1]
            if hasattr(opponent, "notify"):
                opponent.notify(board, action)
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
            
            board_history+=board

            
            is_game_over = self.game.getGameEnded(board, curPlayer, it+1)


        for player in players[0], players[2]:
            if hasattr(player, "endGame"):
                player.endGame()

        board_history = (board_history - np.max(board_history))*(-1)
        board_history[board_history == 0] = -1
        board_history[board_history == np.max(board_history)]= 0

        if verbose:
            assert self.game.display
            print("Game over: Turn ", str(it), "Result: Player ", str(curPlayer*is_game_over), " wins!")
            self.game.display(board_history)


        return curPlayer*is_game_over
        
        

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1
        one_on_the_play, two_on_the_draw = oneWon, twoWon
        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1
        one_on_the_draw = oneWon-one_on_the_play
        two_on_the_play = twoWon-two_on_the_draw
        
        result_table = {
        "Player": ["Player 1", "Player 2"],
        "Games Won Going First": [one_on_the_play, two_on_the_play],
        "Games Won Going Second": [one_on_the_draw, two_on_the_draw],
        }

        header = f"{'Player':<12}{'Going First':<18}{'Going Second':<15}"
        separator = "-" * len(header)
        rows = [
            f"{result_table['Player'][i]:<12}{result_table['Games Won Going First'][i]:<18}{result_table['Games Won Going Second'][i]:<15}"
            for i in range(len(result_table["Player"]))
        ]

        print(header)
        print(separator)
        print("\n".join(rows))

        return oneWon, twoWon, draws
