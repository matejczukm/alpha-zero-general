import numpy as np
from huggingface_hub import hf_hub_download

from HexGame.HexGame import *
from HexGame.pytorch.NNet import NNetWrapper as HexNet
from TriangleGame.pytorch.NNet import NNetWrapper as TriNet
from TriangleGame.TriangleGame import *
from PackitMCTS import MCTS
from utils import *
import logging
import os
import torch

log = logging.getLogger(__name__)


class AIPlayer:

    def __init__(self, size, mode='triangular', weights_filename = 'best_cpuct_1.pth.tar'):

        mcts_args = dotdict({'numMCTSSims': 10,
                             'cpuct': 1.0})

        args = dotdict({'tri_weights_folder': './alpha-zero-general/packit-polygons-models/pytorch/triangle_models/',
                        'hex_weights_folder': './alpha-zero-general/packit-polygons-models/pytorch/hex_models/',
                        'hex_hf_weights_path': 'pytorch/hex_models/',
                        'tri_hf_weights_path': 'pytorch/triangle_models/',
                        'hf_repo_id': 'lgfn/packit-polygons-models'})
        self.weights_filename = weights_filename
        if mode == 'triangular':
            self.game = TriangleGame(size)
            args.tri_weights_folder += 'size_' + str(size) + '/'

            self.nnet = TriNet(self.game)

            weights_path = args.tri_hf_weights_path + 'size_' + str(size) + '/' + self.weights_filename

            try:
                print('Downloading weights...')
                print('Accessing ', weights_path)

                weights_hf = hf_hub_download(repo_id=args.hf_repo_id, filename=weights_path)

                map_location = None if torch.cuda.is_available() else 'cpu'
                checkpoint = torch.load(weights_hf, map_location=map_location, weights_only=True)
                self.nnet.nnet.load_state_dict(checkpoint['state_dict'])
                print('Download successful')

            except Exception as e:
                print(e)
                print('Weights not found, using untrained model')



        elif mode == 'hexagonal':
            self.game = HexGame(size)
            args.hex_weights_folder += 'size_' + str(size) + '/'
            self.nnet = HexNet(self.game)

            weights_path = args.hex_hf_weights_path + 'size_' + str(size) + '/' + self.weights_filename

            try:
                print('Downloading weights...')
                print('Accessing ', weights_path)
                weights_hf = hf_hub_download(repo_id=args.hf_repo_id, filename=weights_path)

                map_location = None if torch.cuda.is_available() else 'cpu'
                checkpoint = torch.load(weights_hf, map_location=map_location, weights_only=True)
                self.nnet.nnet.load_state_dict(checkpoint['state_dict'])

                print('Donwload successful')

            except Exception as e:
                print(e)
                print("Weights not found, using untrained model")

        else:
            raise Exception('Invalid mode')

        self.mcts = MCTS(self.game, self.nnet, mcts_args)

        return

    def mcts_get_action(self, board, turn):
        """
        Returns np.array representation of model's action using mcts simulations
        """

        valids = self.game.getValidMoves(board, 1, turn)
        if np.max(valids) == 0:
            return np.zeros_like(board)
        probs = self.mcts.getActionProb(board, turn, temp=0)
        if np.max(probs * valids) == 0:
            log.info('Returning random move')
            action_ix = np.random.choice(np.nonzero(valids)[0])
            return self.game.action_space[action_ix]
        action_ix = np.argmax(probs * valids)
        return self.game.action_space[action_ix]

    def nnet_get_action(self, board, turn):
        """
        Returns np.array representation of model's action using only neural net predicition
        """
        valids = self.game.getValidMoves(board, 1, turn)
        if np.max(valids) == 0:
            return np.zeros_like(board)
        probs, v = self.nnet.predict(board)
        if np.max(probs * valids) == 0:
            log.info('Returning random move')
            action_ix = np.random.choice(np.nonzero(valids)[0])
            return self.game.action_space[action_ix]
        action_ix = np.argmax(probs * valids)
        return self.game.action_space[action_ix]

    def get_action_for_arena(self, board, turn):
        valids = self.game.getValidMoves(board, 1, turn)
        if np.max(valids) == 0:
            return np.zeros_like(board)
        probs = self.mcts.getActionProb(board, turn, temp=0)
        if np.max(probs * valids) == 0:
            log.info('Returning random move')
            action_ix = np.random.choice(np.nonzero(valids)[0])
            return self.game.action_space[action_ix]
        action_ix = np.argmax(probs * valids)
        return action_ix
