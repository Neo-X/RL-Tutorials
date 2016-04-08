import random
import numpy as np
import math
import cPickle
import json
import os
import sys

# Networks
from RLLogisticRegression import RLLogisticRegression
from NeuralNet import NeuralNet 
from RLNeuralNetwork import RLNeuralNetwork
from RLNeuralNetworkDQ import RLNeuralNetworkDQ
from RLDeepNet import RLDeepNet 
from DeepCACLA import DeepCACLA
from DeepDPG import DeepDPG 
from ForwardDynamicsNetwork import ForwardDynamicsNetwork

# Games
from MapGame import Map
from BallGame1D import BallGame1D
from BallGame1DFuture import BallGame1DFuture

from RL_visualizing import *
from RLVisualize import RLVisualize
from NNVisualize import NNVisualize
from ExperienceMemory import ExperienceMemory

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
    
    # make a color map of fixed colors
    #try: 
        file = open(sys.argv[1])
        settings = json.load(file)
        file.close()
        map = loadMap()
        # Normalization constants for data
        max_reward = settings['max_reward']
        # max_reward = 1.0
        max_state = settings['max_state']
        
        print "Max Reward: " + str(max_reward)
        print "Max State: " + str(max_state)
        
        
        # game = Map(map)
        game = None
        game_type = settings['game_type']
        if game_type == 'BallGame1DFuture':
            game = BallGame1DFuture()
        elif game_type == 'BallGame1D':
            game = BallGame1D()
        else:
            print "Unrecognized game: " + str(game_type)
            sys.exit()
            
        action_bounds = np.array(settings['action_bounds'])
        action_length = len(action_bounds[0])
        data_folder = settings['data_folder']
        states = np.array([max_state])
        state_length = len(max_state)
        action_space_continuous=False
        
        file_name=data_folder+"navigator_agent_"+str(settings['agent_name'])+".pkl"
        model = cPickle.load(open(file_name))
        
        forwardDynamicsModel = ForwardDynamicsNetwork(state_length=state_length,action_length=action_length)
         
        if action_space_continuous:
            # X, Y, U, V, Q = get_continuous_policy_visual_data(model, max_state, game)
            X, Y, U, V, Q = get_continuous_policy_visual_data1D(model, max_state, game)
        else:
            X, Y, U, V, Q = get_policy_visual_data(model, max_state, game)
        game.init(U, V, Q)
        
        
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        bellman_errors = []
        reward_over_epocs = []
        values = []
        step=0
        num_actions = 10
        scaling = 1.0
        
        actions = (np.random.rand(num_actions,1)-0.5) * 2.0 * scaling
        for action in actions:
            # ballGame.resetTarget()
            state = game.getState()
            print "State: " + str(state)
            print "Action: " + str(action)
            pa = model.predict([norm_state(state, max_state)])
            if action_space_continuous:
                action = scale_action(pa, action_bounds)
                reward = game.actContinuous(action)
            elif not action_space_continuous:
                reward = game.act(action)
            print "Reward: " + str(reward)
            
            

    #except Exception, e:
    #    print "Error: " + str(e)
    #    raise e
