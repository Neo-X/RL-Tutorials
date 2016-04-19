import matplotlib.pyplot as plt
# from matplotlib import mpl
import numpy as np
# import matplotlib.animation as animation
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
    return norm_action(action_=state, action_bounds_=max_state)

def scale_state(state, max_state):
    return scale_action(normed_action_=state, action_bounds_=max_state)

def norm_state2(state, max_state):
    """
    For when the origin is in the centre of the environment
    """
    return (state)/max_state

def clampAction(actionV, bounds):
    """
    bounds[0] is lower bounds
    bounds[1] is upper bounds
    """
    for i in range(len(actionV)):
        if actionV[i] < bounds[0][i]:
            actionV[i] = bounds[0][i]
        elif actionV[i] > bounds[1][i]:
            actionV[i] = bounds[1][i]
    return actionV

def norm_action(action_, action_bounds_):
    """
        
        Normalizes the action 
        Where the middle of the action bounds are mapped to 0
        upper bound will correspond to 1 and -1 to the lower
        from environment space to normalized space
    """
    avg = (action_bounds_[0] + action_bounds_[1])/2
    return (action_ - (avg)) / (action_bounds_[1]-avg)

def scale_action(normed_action_, action_bounds_):
    """
        from normalize space back to environment space
        Normalizes the action 
        Where 0 in the action will be mapped to the middle of the action bounds
        1 will correspond to the upper bound and -1 to the lower
    """
    avg = (action_bounds_[0] + action_bounds_[1])/2.0
    return normed_action_ * (action_bounds_[1] - avg) + avg

def get_policy_visual_data(model_, max_state, game):
    X,Y = np.mgrid[0:16,0:16]
    U = []
    V = []
    Q = []
    for i in range(16):
        t_u = []
        t_v = []
        t_q = []
        for j in range(16):
            state = np.array([X[i][j],Y[i][j]])
            pa = model_.predict([norm_state(state,max_state)])
            # pa = model_.predict([norm_state(state,max_state)])
            q = np.max(model_.q_values([norm_state(state,max_state)]))
            # q=0 
            move = game.move(pa)
            t_u.append(move[0])
            t_v.append(move[1])
            t_q.append(q)
        U.append(t_u)
        V.append(t_v)
        Q.append(t_q)
    
    U = np.array(U)*1.0
    V = np.array(V)*1.0
    return (X, Y, U, V, Q)


def get_continuous_policy_visual_data(model_, max_state, game):
    X,Y = np.mgrid[0:16,0:16]
    U = []
    V = []
    Q = []
    for i in range(16):
        t_u = []
        t_v = []
        t_q = []
        for j in range(16):
            state = np.array([X[i][j],Y[i][j]])
            pa = model_.predict([norm_state(state,max_state[:2])])
            # pa = model_.predict([norm_state(state,max_state)])
            q = (model_.q_value([norm_state(state,max_state[:2])]))
            # q=0 
            move = pa
            t_u.append(move[0])
            t_v.append(move[1])
            t_q.append(q)
        U.append(t_u)
        V.append(t_v)
        Q.append(t_q)
    
    U = np.array(U)*1.0
    V = np.array(V)*1.0
    return (X, Y, U, V, Q)

def get_continuous_policy_visual_data1D(model_, state_bounds, game):
    """
        For policies with one output action parameter
    """
    size_=16
    X,Y = game.getGrid()
    U = np.array(X)
    V = np.array(X)
    Q = np.array(X)
    for i in range(size_):
        for j in range(size_):
            if len(state_bounds[0]) == 4:
                state = np.array([X[i][j],Y[i][j], X[i][j], X[i][j]])
            elif len(state_bounds[0]) == 2:
                state = np.array([X[i][j],Y[i][j]])
            pa = model_.predict([norm_state(state,state_bounds)])
            # pa = model_.predict([norm_state(state,max_state)])
            q = (model_.q_value([norm_state(state,state_bounds)]))
            # q=0 
            move = pa
            U[i,j]=move[0]
            V[i,j]=0
            Q[i,j]=q
    
    U = np.array(U)*1.0
    V = np.array(V)*1.0
    Q = np.array(Q)
    # Switch X and Y?
    return (X, Y, U, V, Q)

class RLVisulize(object):
    
    def __init__(self, map):
        # self._fig, (self._map_ax, self._policy_ax) = plt.subplots(1, 2, sharey=False)
        self._map = map
        self._bounds = np.array([[0,0], [15,15]])

    def final_policy(self, X, Y, U, V, Q):
        X,Y = np.mgrid[0:self._bounds[1][0]+1,0:self._bounds[1][0]+1]
        # print X,Y
        # print U,V
        # print Q
        fig, ax = plt.subplots(1)
        # self._policy = self._policy_ax.quiver(X[::2, ::2],Y[::2, ::2],U[::2, ::2],V[::2, ::2], linewidth=0.5, pivot='mid', edgecolor='k', headaxislength=5, facecolor='None')
        ax.quiver(X,Y,U,V,Q, alpha=.75, linewidth=1.0, pivot='mid', angles='xy', linestyles='-', scale=25.0)
        ax.quiver(X,Y,U,V, linewidth=0.5, pivot='mid', edgecolor='k', headaxislength=3, facecolor='None', angles='xy', linestyles='-', scale=25.0)
        plt.title("Final Policy")
        # these are matplotlib.patch.Patch properties
        textstr = """$\max q=%.2f$\n$\min q=%.2f$"""%(np.max(Q), np.min(Q))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
        
        # place a text box in upper left in axes coords
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        plt.show()
        
        
        
        
if __name__ == "__main__":
    
    max_state = 8.0
    file_name=sys.argv[1]
    model = cPickle.load(open(file_name)) 
    map = loadMap()
    rlv = RLVisulize(map)
    game = Map(map)
    X, Y, U, V, Q = get_policy_visual_data(model, max_state, game)
    rlv.final_policy(X, Y, U, V, Q)