from MapGame import Map
from BallGame import BallGame
import random
import numpy as np
import math
import cPickle
import json
import os

from RLLogisticRegression import RLLogisticRegression
from NeuralNet import NeuralNet 
from RLNeuralNetwork import RLNeuralNetwork
from RLNeuralNetworkDQ import RLNeuralNetworkDQ
from RLDeepNet import RLDeepNet 
from DeepCACLA import DeepCACLA
from DeepDPG import DeepDPG 
import sys

from RL_visualizing import *
from RLVisualize import RLVisualize
from ExperienceMemory import ExperienceMemory

import matplotlib.pyplot as plt


def eGreedy(pa1, ra2, e):
    """
        epsilon greedy action select
        pa1 is best action from policy
        ra1 is the random action
        e is proabilty to select random action
        0 <= e < 1.0
    """
    r = random.random()
    if r < e:
        return ra2
    else:
        return pa1
    
def eOmegaGreedy(pa1, ra1, ra2, e, omega):
    """
        epsilon greedy action select
        pa1 is best action from policy
        ra1 is the noisy policy action action
        ra2 is the random action
        e is proabilty to select random action
        0 <= e < omega < 1.0
    """
    r = random.random()
    if r < e:
        return ra2
    elif r < omega:
        return ra1
    else:
        return pa1
    
def randomExporation(explorationRate, actionV):
    out = []
    for i in range(len(actionV)):
        out.append(actionV[i] + random.gauss(actionV[i], explorationRate))
    return out

def randomUniformExporation(bounds):
    out = []
    for i in range(len(bounds[0])):
        out.append(np.random.uniform(bounds[0][i],bounds[1][i],1)[0])
    return out


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
    
def collectExperienceActionsContinuous(experience, action_bounds):
    i = 0
    while i < experience.history_size():
        game.reset()
        t=0
        while not game.reachedTarget():
            if (t > 31):
                game.reset()
                t=0
                
            state = game.getState()
            action = game.move(random.choice(action_selection))
            # randomAction = randomUniformExporation(action_bounds) # Should select from 8 original actions
            # action = clampAction(randomAction, action_bounds)
            reward = game.actContinuous(action)
            resultState = game.getState()
            # tup = ExperienceTuple(state, [action], resultState, [reward])
            # Everything should be normalized to be between -1 and 1
            reward_ = (reward+(max_reward/2.0))/(max_reward*0.5)
            # reward_ = (reward)/(max_reward)
            # reward_ = (reward+max_reward)/(max_reward)
            experience.insert(norm_state(state, max_state), [action], norm_state(resultState, max_state), [reward_])
            i+=1
            t+=1

    print "Done collecting experience from " + str(experience.samples()) + " samples."
    return experience  

    
if __name__ == "__main__":
    
    file = open(sys.argv[1])
    settings = json.load(file)
    file.close()
    
    epsilon = 0.45 # It is important to have some space between these values especially now that the experience buffer starts loaded with random actions
    omega = 0.8
    p = 1.0
    max_reward = math.sqrt(16**2 * 2) + 5.0
    
    game = BallGame()
    game.init([],[],[])
    action_space_continuous = True
    max_training_steps = 10
    action_bounds = settings['action_bounds']
    action_bounds = [[-0.8,0.1],[0.8,1.0]]
    data_folder = settings['data_folder']
    num_actions=8
    action_selection = range(num_actions)
    
    init_state=[]
    result_state=[]
    action_path=[]
    
    action=0
    reward=0
    step=0
    while step < max_training_steps:
        game.reset()
        step+=1
            
        state = game.getState()
        init_state.append(state)
        # pa = model.predict([norm_state(state, max_state)])
        pa = [0,0]
        
        action = randomUniformExporation(action_bounds) # Completely random action
        print action
        action = [1,1]
        reward = game.actContinuous(action)
            
        resultState = game.getState()
        # tup = ExperienceTuple(state, [action], resultState, [reward])
        # Everything should be normalized to be between -1 and 1
        reward_ = (reward+(max_reward/2.0))/(max_reward*0.5)
        
        result_state.append(resultState)
        
    game.finish()
    init_state = np.array(init_state)
    result_state = np.array(result_state)
    print init_state, result_state
    plt.plot(init_state[:,0],init_state[:,1],'bo', ms=6)
    plt.plot(result_state[:,0],result_state[:,1],'ro', ms=6)
    

    plt.show()
