import matplotlib.pyplot as plt
from matplotlib import mpl
import numpy as np
import matplotlib.animation as animation
import random
import cPickle
import sys
from MapGame import Map


def loadMap():
    dataset="map.json"
    import json
    dataFile = open(dataset)
    s = dataFile.read()
    data = json.loads(s)
    dataFile.close()
    return data["map"]

def norm_state(state, max_state):
    return (state-max_state)/max_state

def get_policy_visual_data(model_, max_state, game):
    X,Y = np.mgrid[0:16,0:16]
    U = []
    V = []
    for i in range(16):
        t_u = []
        t_v = []
        for j in range(16):
            state = np.array([X[i][j],Y[i][j]])
            pa = model_.predict([norm_state(state,max_state)])[0]
            move = game.move(pa)
            t_u.append(move[0])
            t_v.append(move[1])
        U.append(t_u)
        V.append(t_v)
    
    U = np.array(U)*2.0
    V = np.array(V)*2.0
    return (X, Y, U,V)

class RLVisulize(object):
    
    def __init__(self, map):
        # self._fig, (self._map_ax, self._policy_ax) = plt.subplots(1, 2, sharey=False)
        self._map = map
        self._bounds = np.array([[0,0], [15,15]])

    def final_policy(self, X, Y, U, V):
        X,Y = np.mgrid[0:self._bounds[1][0]+1,0:self._bounds[1][0]+1]
        print X,Y
        # self._policy = self._policy_ax.quiver(X[::2, ::2],Y[::2, ::2],U[::2, ::2],V[::2, ::2], linewidth=0.5, pivot='mid', edgecolor='k', headaxislength=5, facecolor='None')
        plt.quiver(X,Y,U,V, linewidth=1.0, pivot='mid', edgecolor='k', headaxislength=3, facecolor='black', angles='xy', linestyles='-', scale=50.0)
        plt.show()
        
        
        
        
if __name__ == "__main__":
    
    max_state = 8.0
    file_name=sys.argv[1]
    model = cPickle.load(open(file_name)) 
    map = loadMap()
    rlv = RLVisulize(map)
    game = Map(map)
    X, Y, U, V = get_policy_visual_data(model, max_state, game)
    rlv.final_policy(X, Y, U, V)