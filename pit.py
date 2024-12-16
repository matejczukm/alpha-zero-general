import PackitArena
from PackitMCTS import MCTS
from HexGame.HexGame import HexGame
from HexGame.HexPlayers import RandomPlayer
from HexGame.keras.NNet import NNetWrapper as HexKerasNNet
from huggingface_hub import hf_hub_download


import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
hf_weights_path = 'keras/hex_models/'
weights_filename = 'best.weights.h5'
hf_repo_id = 'lgfn/packit-polygons-models'

random_vs_cpu = False
from_hf = True
size = 3

g = HexGame(size)

# all players
rp = RandomPlayer(g).play
# gp = GreedyOthelloPlayer(g).play
# hp = HumanOthelloPlayer(g).play



# nnet players
n1 = HexKerasNNet(g)
# if mini_othello:
#     n1.load_checkpoint('./pretrained_models/othello/pytorch/','6x100x25_best.pth.tar')
# else:
n1.load_checkpoint('./packit-polygons-models/hex_models/size_'+str(size)+'/','best.weights.h5')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
# n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
n1p = lambda x, t: np.argmax(mcts1.getActionProb(x, t, temp=0))


if random_vs_cpu:
    # player2 = hp
    player2 = rp
elif from_hf:
    weights_path = hf_weights_path + 'size_' + str(size) + '/' + weights_filename

    weights_path = hf_hub_download(repo_id=hf_repo_id, filename=weights_path)
    # HexKerasNNet.nnet.model.load_weights(weights_path)
    n2 = HexKerasNNet(g)
    n2.nnet.model.load_weights(weights_path)
    args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, args2)
    # n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))
    n2p = lambda x, t: np.argmax(mcts2.getActionProb(x, t, temp=0))

    player2 = n2p
else:
    n2 = HexKerasNNet(g)
    n2.load_checkpoint('./hex_temp/', 'best.weights.h5')
    args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, args2)
    # n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))
    n2p = lambda x, t: np.argmax(mcts2.getActionProb(x, t, temp=0))


    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

arena = PackitArena.Arena(n1p, player2, g, display=HexGame.display)

print(arena.playGames(100, verbose=0))
