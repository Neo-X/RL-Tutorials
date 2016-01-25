
import theano
from theano import tensor as T
import numpy as np
import random


class ExperienceMemory(object):
    
    def __init__(self, state_length, action_length, memory_length):
        
        
        self._history_size=memory_length
        self._history_update_index=0 # where the next experience should write
        self._inserts=0
        self._state_length = state_length
        self._action_length = action_length
        
        # self._state_history = theano.shared(np.zeros((self._history_size, state_length)))
        # self._action_history = theano.shared(np.zeros((self._history_size, action_length)))
        # self._nextState_history = theano.shared(np.zeros((self._history_size, state_length)))
        # self._reward_history = theano.shared(np.zeros((self._history_size, 1)))
        
        self._state_history = (np.zeros((self._history_size, state_length)))
        self._action_history = (np.zeros((self._history_size, action_length)))
        self._nextState_history = (np.zeros((self._history_size, state_length)))
        self._reward_history = (np.zeros((self._history_size, 1)))
        
    def insert(self, state, action, nextState, reward):
        self._inserts+=1
        self._history_update_index+=1
        if ( (self._history_update_index % (self._history_size-1) ) == 0):
            self._history_update_index=0
        # print "Tuple: " + str(state) + ", " + str(action) + ", " + str(nextState) + ", " + str(reward)
        self._state_history[self._history_update_index] = np.array(state)
        self._action_history[self._history_update_index] = np.array(action)
        self._nextState_history[self._history_update_index] = np.array(nextState)
        self._reward_history[self._history_update_index] = np.array(reward)
        
    def samples(self):
        return self._inserts
        
    def get_batch(self, batch_size=32):
        """
        len(experience > batch_size
        """
        # indices = list(nprnd.randint(low=0, high=len(experience), size=batch_size))
        indices = (random.sample(range(0, self._history_size), batch_size))
        # print indices

        state = []
        action = []
        resultState = []
        reward = []
        
        for i in indices:
            state.append(self._state_history[i])
            action.append(self._action_history[i])
            resultState.append(self._nextState_history[i])
            reward.append(self._reward_history[i])
        # print c
        # print experience[indices]
        state = np.array(state)
        if (self._action_length > 1):
            action = np.array(action)
        else:
            action = np.array(action, dtype='int32')
        resultState = np.array(resultState)
        reward = np.array(reward)
         
        return (state, action, resultState, reward)
        