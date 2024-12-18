import logging

import coloredlogs
import sys
sys.path.insert(0, r"C:\Users\User\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages")
#I really should fix that
from PackitCoach import Coach
from HexGame.HexGame import HexGame
from TriangleGame.TriangleGame import TriangleGame
from HexGame.keras.NNet import NNetWrapper as keras_hexnnet
from HexGame.keras.NNetSmall import NNetWrapper as keras_hexnnetsm
from TriangleGame.keras.NNet import NNetWrapper as keras_trinnet
from TriangleGame.keras.NNetSmall import NNetWrapper as keras_trinnetsm

from HexGame.pytorch.NNet import NNetWrapper as torch_hexnnet
from TriangleGame.pytorch.NNet import NNetWrapper as torch_trinnet
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 3,             # Number of iterations
    'numEps': 25,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 30,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1.2,
    'checkpoint': './packit-polygons-models/',
    'load_model': False,
    'load_folder_file': ('./packit-polygons-models/pytorch/','best_cpuct_1.pth.tar'),
    'best_filename': 'best',
    'numItersForTrainExamplesHistory': 20,
    'tri_board_sizes': [],
    'hex_board_sizes': [2],
    'model': 'pytorch' #either pytorch or keras
})


def main():


    for size in args.tri_board_sizes:

        log.info('Loading %s...', TriangleGame.__name__)
        g = TriangleGame(size)



        if args.model == 'keras':
            log.info('Loading %s...', keras_trinnet.__name__)
            if size < 5:
                
                nnet = keras_trinnetsm(g)
            else:
                nnet = keras_trinnet(g)
        elif args.model == 'pytorch':
            log.info('Loading %s...', torch_trinnet.__name__)
            nnet = torch_trinnet(g)
        else:
            log.error('Incorrect model argument, choose pytorch or keras')
            return

        
        args.load_folder_file = ('./packit-polygons-models/' + args.model + '/triangle_models/size_'+str(size), 'best.pth.tar')

        if args.load_model:
            log.info('Loading checkpoint "%s%s"...', args.load_folder_file[0], args.load_folder_file[1])
            nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
        else:
            log.warning('Not loading a checkpoint!')

        log.info('Loading the Coach...')
        args.checkpoint = './packit-polygons-models/' + args.model + '/triangle_models/size_'+str(size)+'/'
        c = Coach(g, nnet, args)

        if args.load_model:
            log.info("Loading 'trainExamples' from file...")
            c.loadTrainExamples()

        log.info('Starting the learning process for board of size %s 🎉', size)
        c.learn()




    for size in args.hex_board_sizes:
        log.info('Loading %s...', HexGame.__name__)
        g = HexGame(size)


        if args.model == 'keras':
            log.info('Loading %s...', keras_hexnnet.__name__)
            
            if size < 3:
                
                nnet = keras_hexnnetsm(g)
            else:
                nnet = keras_hexnnet(g)

        elif args.model == 'pytorch':
            log.info('Loading %s...', torch_hexnnet.__name__)
            nnet = torch_hexnnet(g)
        else:
            log.error('Incorrect model argument, choose pytorch or keras')
            return
        
        args.load_folder_file = ('./packit-polygons-models/' + args.model + '/triangle_models/size_'+str(size), 'best.pth.tar')


        if args.load_model:
            log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
            nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
        else:
            log.warning('Not loading a checkpoint!')

        log.info('Loading the Coach...')
        args.checkpoint = './packit-polygons-models/' + args.model + '/hex_models/size_'+str(size)+'/'
        c = Coach(g, nnet, args)

        if args.load_model:
            log.info("Loading 'trainExamples' from file...")
            c.loadTrainExamples()

        log.info('Starting the learning process for board of size %s 🎉', size)
        c.learn()

if __name__ == "__main__":
    main()
