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
    
    def selectTarget(self, state ):
        _target = self._targetSelector.predict(state)
    
    def getTargetAction(self, _target, state):
        _target = self._targetSelector.predict(state)
        print "Inner State: " + str(state)
        #change the state
        _state = np.zeros_like(state)
        _state[0][0] = state[0][0]
        __state = np.zeros(len(state[0])-1)
        __state[_target] = 1.0
        
        _state[0][1:] = __state
        return self._actionNetwork.predict(_state)
        
    def predict(self, state):
        _target = self._targetSelector.predict(state)
        print "Inner State: " + str(state)
        #change the state
        _state = np.zeros_like(state)
        _state[0][0] = state[0][0]
        __state = np.zeros(len(state[0])-1)
        __state[_target] = 1.0
        
        _state[0][1:] = __state
        return self._actionNetwork.predict(_state)
        
        # _targetSelector to get state
        # actionNetwork to get the parameters for the action
        
    
    def q_value(self, state):
        # _targetSelector to get state
        # actionNetwork to get the parameters for the action
        return self._targetSelector.q_value(state)
    
    def bellman_error(self, state, action, reward, result_state):
        # bellman error for _target_selector
        return self._targetSelector.bellman_error(state, action, reward, result_state)
    
    




