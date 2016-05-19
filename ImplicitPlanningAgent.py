import theano
from theano import tensor as T
import numpy as np
import lasagne

from DeepCACLA import DeepCACLA
from RLDeepNet import RLDeepNet


# For debugging
# theano.config.mode='FAST_COMPILE'

class ImplicitPlanningAgent(object):
    
    def __init__(self, n_in, n_out, actionNetwork):
        
        self._targetSelector = RLDeepNet(n_in=n_in, n_out=n_out)
        # _actionNetwork = RLDeepNet(n_in=n_in, n_out=n_out) # should load this network from a file
        self._actionNetwork = actionNetwork
        
        
    def train(self, states, actions, rewards, result_states):
        self._targetSelector.train(states, actions, rewards, result_states)
    
    def predict(self, state):
        pass
    
    def q_value(self, state):
        pass
    
    def bellman_error(self, state, action, reward, result_state):
        pass
    
    




