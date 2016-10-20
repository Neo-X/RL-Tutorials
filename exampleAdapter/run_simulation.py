
import sys

sys.path.append("../")
import deap
import json
import random
import numpy as np
import copy
import example
import numpy as np
import math
import matplotlib.pyplot as plt
import sys

if (__name__ == "__main__"):
    
    word = example.Word("blah blah")
    print ("The Word is: ", word.getWord())
    
    function = example.Function()

    state_bounds = np.array([[0.0],[5.0]])
    action_bounds = np.array([[-4.0],[2.0]])
    function.setStateBounds(state_bounds)
    function.setActionBounds(action_bounds)
    experience_length = 200
    batch_size=32
    # states = np.repeat(np.linspace(0.0, 5.0, experience_length),2, axis=0)
    states = np.linspace(0.0, 5.0, experience_length)
    states = np.reshape(states, (len(states), 1))
    shuffle = range(experience_length)
    states = states[shuffle]
    # random.shuffle(shuffle)
    print ("States: " , states)
    old_states = states
    norm_states=[]
    norm_states = list(map(function.norm_state, states))
    X_test = X_train = norm_states
    
    # states = np.linspace(-5.0,-2.0, experience_length/2)
    # states = np.append(states, np.linspace(-1.0, 5.0, experience_length/2))
    # print states
    # actions = np.array(map(fNoise, states))
    actions = np.reshape(np.array(list(map(function.func, states))), (len(states), 1))
    # actions = actions[shuffle]
    print ("Actions: ", actions)
    norm_actions = np.array(list(map(function.norm_action, actions)))
    y_train = y_test = norm_actions
    settings = {}
    print("X_test:", X_test)
    print("Y_test:", y_test)