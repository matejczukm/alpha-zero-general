import logging

import coloredlogs

from PackitCoach import Coach
from HexGame.HexGame import HexGame
from TriangleGame.TriangleGame import TriangleGame
from HexGame.keras.NNet import NNetWrapper as hexnnet
from HexGame.keras.NNetSmall import NNetWrapper as hexnnetsm
from TriangleGame.keras.NNet import NNetWrapper as trinnet
from TriangleGame.keras.NNetSmall import NNetWrapper as trinnetsm
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 50,             # Number of iterations
    'numEps': 25,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 30,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'checkpoint': './packit-polygons-models/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
    'tri_board_sizes': [3,4,5],
    'hex_board_sizes': [3,4,5]
})


def main():
    for size in args.tri_board_sizes:

        log.info('Loading %s...', TriangleGame.__name__)
        g = TriangleGame(size)
        
        log.info('Loading %s...', trinnet.__name__)
        if size < 5:
            
            nnet = trinnetsm(g)
        else:
            nnet = trinnet(g)
        

        # if args.load_model:
        #     log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        #     nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
        # else:
        #     log.warning('Not loading a checkpoint!')

        log.info('Loading the Coach...')
        args.checkpoint = './packit-polygons-models/triangle_models/size_'+str(size)+'/'
        c = Coach(g, nnet, args)

        # if args.load_model:
        #     log.info("Loading 'trainExamples' from file...")
        #     c.loadTrainExamples()

        log.info('Starting the learning process for board of size %s 🎉', size)
        c.learn()




    for size in args.hex_board_sizes:

        log.info('Loading %s...', HexGame.__name__)
        g = HexGame(size)
        
        log.info('Loading %s...', hexnnet.__name__)
        if size < 3:
            nnet = hexnnetsm(g)
        else:
            nnet = hexnnet(g)
        

        # if args.load_model:
        #     log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        #     nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
        # else:
        #     log.warning('Not loading a checkpoint!')

        log.info('Loading the Coach...')
        args.checkpoint = './packit-polygons-models/hex_models/size_'+str(size)+'/'
        c = Coach(g, nnet, args)

        # if args.load_model:
        #     log.info("Loading 'trainExamples' from file...")
        #     c.loadTrainExamples()

        log.info('Starting the learning process for board of size %s 🎉', size)
        c.learn()

if __name__ == "__main__":
    main()
