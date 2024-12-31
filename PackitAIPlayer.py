import numpy as np
from huggingface_hub import hf_hub_download

from HexGame.HexGame import *
from HexGame.pytorch.NNet import NNetWrapper as torch_hexnnet
from TriangleGame.pytorch.NNet import NNetWrapper as torch_trinnet
from TriangleGame.TriangleGame import *
from PackitMCTS import MCTS
from utils import *
import logging
import os
import torch


class AIPlayer:
    log = logging.getLogger(__name__)

    def __init__(self,
                 size,
                 mode,
                 model_framework = 'pytorch',
                 local = False,
                 local_folder=None,
                 local_filename = None,
                 hf_repo_id='lgfn/packit-polygons-models',
                 hf_filename = None,
                 numMCTSSims = 50,
                 cpuct = 1,
                 nnet = None):
        
        assert mode == 'triangular' or mode == 'hexagonal', "Invalid game mode, choose 'triangular' or 'hexagonal'"
        assert model_framework == 'pytorch' or model_framework == 'keras', "Invalid model argument, choose 'pytorch' or 'keras'"
        assert isinstance(local, bool), "'local' should be of boolean type"
        
        self.size = size
        self.mode = mode
        self.model_framework = model_framework
        mcts_args = dotdict({'numMCTSSims': numMCTSSims,
                        'cpuct': cpuct})

        

        if mode == 'triangular':
            self.game = TriangleGame(size)
        else:
            self.game = HexGame(size)

        if nnet:
            self.nnet = nnet
            self.mcts = MCTS(self.game, self.nnet, mcts_args)
            return

        if model_framework == 'keras':
            print('work on keras models in progress')
        else:
            if mode == 'triangular':
                self.nnet = torch_trinnet(self.game)
            else:
                self.nnet = torch_hexnnet(self.game)

        mcts_args = dotdict({'numMCTSSims': numMCTSSims,
                        'cpuct': cpuct})
        self.mcts = MCTS(self.game, self.nnet, mcts_args)
        
        if local:
            if not local_folder:
                self.log.error("'local_folder' argument not provided with 'local' being True, initializing with random model")
                return 
            if not local_filename:
                self.log.error("'local_filename' argument not provided with 'local' being True, initializing with random model")
                return 
            filepath = os.path.join(local_folder, local_filename)
            if not os.path.isfile(filepath):
                self.log.error('File ' + filepath + ' does not exist, , initializing with random model')
                return
            self.log.info('Loading local model checkpoint "%s"...', filepath)
            try:
                self.nnet.load_checkpoint(folder = local_folder, filename=local_filename)
                self.log.info('Loading done!')
            except Exception as e:
                self.log.error('Loading failed: ', exc_info=True)
                self.log.warning('Initializing with random model')
                return
        
        else:
            if not hf_filename:
                self.log.info("'hf_filename' argument not provided, using default location")
                if self.mode == 'triangular':
                    mode_str = 'triangle_models'
                else: mode_str = 'hex_models'

                if self.model_framework == 'keras':
                    file = 'best.weights.h5'
                else:
                    file = 'best_cpuct_5.pth.tar'
                hf_filename = self.model_framework + '/' + mode_str + '/size_' + str(self.size) + '/' + file
            
            self.log.info('Loading HuggingFace model: ' + hf_repo_id +'/' + hf_filename +  '...')
            if self.model_framework == 'keras':
                print('keras model loading in the works, aborting')
                return
            else:
                try:
                
                    weights_hf = hf_hub_download(repo_id=hf_repo_id, filename=hf_filename)

                    map_location = None if torch.cuda.is_available() else 'cpu'
                    checkpoint = torch.load(weights_hf, map_location=map_location, weights_only=True)
                    self.nnet.nnet.load_state_dict(checkpoint['state_dict'])
                    self.log.info('Loading done!')
                
                except Exception as e:
                    self.log.error('Loading failed: ', exc_info=True)
                    return

        self.mcts = MCTS(self.game, self.nnet, mcts_args)
        

    def mcts_get_action(self, board, turn):
        """
        Returns np.array representation of model's action using mcts simulations
        """

        valids = self.game.getValidMoves(board, 1, turn)
        if np.max(valids) == 0:
            return np.zeros_like(board)
        probs = self.mcts.getActionProb(board, turn, temp=0)
        if np.max(probs * valids) == 0:
            self.log.info('All valid moves masked, returning random move')
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
            self.log.info('All valid moves masked, returning random move')
            action_ix = np.random.choice(np.nonzero(valids)[0])
            return self.game.action_space[action_ix]
        action_ix = np.argmax(probs * valids)
        return self.game.action_space[action_ix]

    def get_action_for_arena(self, board, turn):
        valids = self.game.getValidMoves(board, 1, turn)
        if np.max(valids) == 0:
            return np.zeros_like(board)
        probs = self.mcts.getActionProb(board, turn, temp=0)
        if np.max(probs*valids)==0:
            self.log.info('All valid moves masked, returning random move')
            action_ix = np.random.choice(np.nonzero(valids)[0])
            return self.game.action_space[action_ix]
        action_ix = np.argmax(probs*valids)
        return action_ix
