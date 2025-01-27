import logging

import coloredlogs

from .PackitCoach import Coach
from .HexGame.HexGame import HexGame
from .TriangleGame.TriangleGame import TriangleGame
# from HexGame.keras.NNet import NNetWrapper as keras_hexnnet
# from HexGame.keras.NNetSmall import NNetWrapper as keras_hexnnetsm
# from TriangleGame.keras.NNet import NNetWrapper as keras_trinnet
# from TriangleGame.keras.NNetSmall import NNetWrapper as keras_trinnetsm

from .HexGame.pytorch.NNet import NNetWrapper as torch_hexnnet
from .TriangleGame.pytorch.NNet import NNetWrapper as torch_trinnet
from .utils import *
import os
from huggingface_hub import hf_hub_download
# from torch import load
# from torch.cuda import is_available
from .PackitAIPlayer import AIPlayer
from pickle import Unpickler
from .PackitNNetWrapper import NNetWrapper



class PackitTrainer:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True)
    log = logging.getLogger(__name__)
    coloredlogs.install(level='INFO')

    def __init__(self, size, mode, model_framework='pytorch', nnet_module=None):

        assert mode == 'triangular' or mode == 'hexagonal', "Invalid game mode, choose 'triangular' or 'hexagonal'"
        assert model_framework == 'pytorch' or model_framework == 'keras', "Invalid model argument, choose 'pytorch' or 'keras'"
        self.size = size
        self.mode = mode
        self.model_framework = model_framework
        self.trainExamplesHistory = []
        self.last_local_folder = None
        self.last_local_filename = None
        self.nnet_module = nnet_module

        if mode == 'triangular':
            self.game = TriangleGame(size)
        else:
            self.game = HexGame(size)

        if model_framework == 'keras':
            print('work on keras models in progress')
        else:
            if self.nnet_module:
                self.nnet = NNetWrapper(self.game, self.nnet_module)
                return
            if mode == 'triangular':
                self.nnet = torch_trinnet(self.game)
            else:
                self.nnet = torch_hexnnet(self.game)

    def loadModel(self, local=False, local_folder=None, local_filename=None, hf_repo_id='lgfn/packit-polygons-models',
                  hf_filename=None):
        self.last_local_folder = local_folder
        self.last_local_filename = local_filename
        assert isinstance(local, bool), "'local' should be of boolean type"
        if local:
            if not local_folder:
                self.log.error("'local_folder' argument not provided with 'local' being True")
                return
            if not local_filename:
                self.log.error("'local_filename' argument not provided with 'local' being True")
                return
            filepath = os.path.join(local_folder, local_filename)
            if not os.path.isfile(filepath):
                self.log.error('File ' + filepath + ' does not exist')
                return
            self.log.info('Loading local model checkpoint "%s"...', filepath)
            try:
                self.nnet.load_checkpoint(folder=local_folder, filename=local_filename)
                self.log.info('Loading done!')
            except Exception as e:
                self.log.error('Loading failed: ', exc_info=True)

        else:
            if not hf_filename:
                self.log.info("'hf_filename' argument not provided, using default location")
                if self.mode == 'triangular':
                    mode_str = 'triangle_models'
                else:
                    mode_str = 'hex_models'

                if self.model_framework == 'keras':
                    file = 'best.weights.h5'
                else:
                    file = 'best_cpuct_5.pth.tar'
                hf_filename = self.model_framework + '/' + mode_str + '/size_' + str(self.size) + '/' + file

            self.log.info('Loading HuggingFace model: ' + hf_repo_id + '/' + hf_filename + '...')
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

    def resetModel(self):
        if self.model_framework == 'keras':
            print('work on keras models in progress')

        if self.nnet_module:
            self.nnet = NNetWrapper(self.game, self.nnet_module)
        else:
            if self.mode == 'triangular':
                self.nnet = torch_trinnet(self.game)
            else:
                self.nnet = torch_hexnnet(self.game)

    def loadTrainingExamples(self, folder, filename):
        filepath = os.path.join(folder, filename)
        if not os.path.isfile(filepath):
            self.log.error('File ' + filepath + ' not found')
            return
        self.log.info("File with trainExamples found. Loading it...")
        with open(filepath, "rb") as f:
            self.trainExamplesHistory = Unpickler(f).load()
        self.log.info('Loading done!')

    def train(self,
              numIters=50,
              numEps=25,
              tempThreshold=15,
              updateThreshold=0.6,
              maxlenOfQueue=200000,
              numMCTSSims=40,
              arenaCompare=40,
              cpuct=5,
              checkpoint_path='',
              best_filename='',
              numItersForTrainExamplesHistory=20
              ):

        if checkpoint_path == '':
            if self.mode == 'triangular':
                checkpoint_path = './packit-polygons-models/' + self.model_framework + '/triangle_models/size_' + str(
                    self.size)
            else:
                checkpoint_path = './packit-polygons-models/' + self.model_framework + '/hex_models/size_' + str(
                    self.size)
        if best_filename == '':
            best_filename = 'best_cpuct_' + str(cpuct) + '.pth.tar'

        args = dotdict({
            'numIters': numIters,
            'numEps': numEps,
            'tempThreshold': tempThreshold,
            'updateThreshold': updateThreshold,
            'maxlenOfQueue': maxlenOfQueue,
            'numMCTSSims': numMCTSSims,
            'arenaCompare': arenaCompare,
            'cpuct': cpuct,
            'checkpoint': checkpoint_path,
            'best_filename': best_filename,
            'numItersForTrainExamplesHistory': numItersForTrainExamplesHistory
        })

        self.log.info('Loading the Coach...')
        c = Coach(self.game, self.nnet, args, trainExamplesHistory=self.trainExamplesHistory,
                  nnet_module=self.nnet_module)
        self.log.info('Starting the learning process for %s board of size %s 🎉', self.mode, self.size)
        c.learn()
        self.last_local_folder = checkpoint_path
        self.last_local_filename = best_filename
        self.loadModel(local=True, local_folder=self.last_local_folder, local_filename=self.last_local_filename)

    def getConfig(self):
        return {
            'size': self.size,
            'mode': self.mode,
            'model_framework': self.model_framework
        }

    def getAIPlayer(self, numMCTSSims=50, cpuct=1):
        return AIPlayer(self.size,
                        mode=self.mode,
                        model_framework=self.model_framework,
                        numMCTSSims=numMCTSSims,
                        cpuct=cpuct,
                        nnet=self.nnet
                        )