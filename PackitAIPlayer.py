import numpy as np
from huggingface_hub import hf_hub_download

from HexGame.HexGame import *
from HexGame.keras.NNet import NNetWrapper as HexNet
from HexGame.keras.NNetSmall import NNetWrapper as HexNetSmall
from TriangleGame.TriangleGame import *
from TriangleGame.keras.NNet import NNetWrapper as TriNet
from TriangleGame.keras.NNetSmall import NNetWrapper as TriNetSmall
import tensorflow as tf
from PackitMCTS import MCTS
from utils import *
import logging
import os

log = logging.getLogger(__name__)


class AIPlayer:

    def __init__(self, size, mode='triangular'):

        mcts_args = dotdict({'numMCTSSims': 10,
                             'cpuct': 1.0})

        args = dotdict({'tri_weights_folder': './packit-polygons-models/triangle_models/',
                        'hex_weights_folder': './packit-polygons-models/hex_models/',
                        'weights_filename': 'best.cpuct_1.pth.tar',
                        # 'tri_hf_weights_path': 'https://huggingface.co/lgfn/packit-polygons-models/resolve/main/triangle_models/',
                        # 'hex_hf_weights_path': 'https://huggingface.co/lgfn/packit-polygons-models/resolve/main/hex_models/',
                        'hex_hf_weights_path': 'pytorch/hex_models/',
                        'tri_hf_weights_path': 'pytorch/triangle_models/',
                        'hf_repo_id': 'lgfn/packit-polygons-models'})

        if mode == 'triangular':
            self.game = TriangleGame(size)
            args.tri_weights_folder += 'size_' + str(size) + '/'

            local_weights_path = args.tri_weights_folder + args.weights_filename

            if size < 5:
                self.nnet = TriNetSmall(self.game)
            else:
                self.nnet = TriNet(self.game)

            if not os.path.isfile(local_weights_path):

                print('Local weights file not found')

                weights_path = args.tri_hf_weights_path + 'size_' + str(size) + '/' + args.weights_filename

                try:
                    print('Downloading weights...')
                    print('Accessing ', weights_path)
                    # self.nnet.nnet.model.load_weights(tf.keras.utils.get_file(args.weights_filename, weights_path))
                    # self.nnet.save_checkpoint(folder=args.tri_weights_folder, filename=args.weights_filename)
                    weights_path = hf_hub_download(repo_id=args.hf_repo_id, filename=weights_path)
                    self.nnet.nnet.model.load_weights(weights_path)
                except Exception as e:
                    print(e)
                    print('Weights not found, using untrained model')


            else:
                self.nnet.load_checkpoint(folder=args.tri_weights_folder, filename=args.weights_filename)


        elif mode == 'hexagonal':
            self.game = HexGame(size)
            args.hex_weights_folder += 'size_' + str(size) + '/'

            local_weights_path = args.hex_weights_folder + args.weights_filename

            if size < 3:
                self.nnet = HexNetSmall(self.game)
            else:
                self.nnet = HexNet(self.game)

            if not os.path.isfile(local_weights_path):

                print('Local weights file not found')

                # weights_path = args.hex_hf_weights_path + 'size_'+str(size)+'/' + args.weights_filename
                weights_path = args.hex_hf_weights_path + 'size_' + str(size) + '/' + args.weights_filename

                try:
                    print('Downloading weights...')
                    print('Accessing ', weights_path)
                    weights_path = hf_hub_download(repo_id=args.hf_repo_id, filename=weights_path)
                    self.nnet.nnet.model.load_weights(weights_path)

                    # self.nnet.nnet.model.load_weights(tf.keras.utils.get_file(args.weights_filename, weights_path))
                    # self.nnet.save_checkpoint(folder=args.hex_weights_folder, filename=args.weights_filename)
                except Exception as e:
                    print(e)
                    print("Weights not found, using untrained model")

            else:
                self.nnet.load_checkpoint(folder=args.hex_weights_folder, filename=args.weights_filename)
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
