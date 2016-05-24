"""
    An interface class for Agents to be used in the system.

"""

class AgentInterface(object):
    
    def __init__(self, n_in, n_out):
        pass        
    
    def train(self, states, actions, rewards, result_states):
        pass
    
    def predict(self, state):
        pass
    
    def q_value(self, state):
        pass
    
    def bellman_error(self, state, action, reward, result_state):
        pass
