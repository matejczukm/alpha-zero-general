import numpy as np
from HexGame.HexGame import *
from HexGame.keras.NNet import NNetWrapper as HexNet
from TriangleGame.TriangleGame import *
from TriangleGame.keras.NNet import NNetWrapper as TriNet
from PackitMCTS import MCTS
from utils import *
import logging

log = logging.getLogger(__name__)
 
class AIPlayer():

    def __init__(self, size, mode = 'tri'):

        args = dotdict({'numMCTSSims': 10, 'cpuct': 1.0})

        #this if clause can be better
        if mode == 'tri':
            self.game = TriangleGame(size)

            #random model for now
            self.nnet = TriNet(self.game)
        else:
            self.game = HexGame(size)

            #random model for now
            self.nnet = HexNet(self.game)
        

        self.mcts = MCTS(self.game,self.nnet, args)

    def mcts_get_action(self, board, turn):
        '''
        Returns np.array representation of model's action using mcts simulations
        '''

        valids = self.game.getValidMoves(board, 1, turn)
        probs = self.mcts.getActionProb(board, turn, temp=0)
        if np.max(probs*valids)==0:
            log.info('Returning random move')
            action_ix = np.random.choice(np.nonzero(valids)[0])
            return self.game.action_space[action_ix]
        action_ix = np.argmax(probs*valids)
        return self.game.action_space[action_ix]


    def nnet_get_action(self, board, turn):
        '''
        Returns np.array representation of model's action using only neural net predicition
        '''
        valids = self.game.getValidMoves(board, 1, turn)
        probs, v = self.nnet.predict(board)
        if np.max(probs*valids)==0:
            log.info('Returning random move')
            action_ix = np.random.choice(np.nonzero(valids)[0])
            return self.game.action_space[action_ix]
        action_ix = np.argmax(probs*valids)
        return self.game.action_space[action_ix]
