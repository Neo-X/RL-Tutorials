from MapGame import Map
import random
import numpy as np
import math
import cPickle
import json

from RLLogisticRegression import RLLogisticRegression
from NeuralNet import NeuralNet 
from ExperienceMemory import ExperienceMemory
from RLNeuralNetwork import RLNeuralNetwork
from RLNeuralNetworkDQ import RLNeuralNetworkDQ
from RLDeepNet import RLDeepNet 
import sys

from RL_visualizing import *

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
    

    
    
if __name__ == "__main__":
    
    # make a color map of fixed colors
    file = open(sys.argv[1])
    settings = json.load(file)
    file.close()
    batch_size=32
    rounds = 1000
    epsilon = 0.8
    map = loadMap()
    # Normalization constants for data
    max_reward = math.sqrt(16**2 * 2) + 5.0
    # max_reward = 1.0
    max_state = 8.0
    
    num_actions=8
    action_selection = range(num_actions)
    
    print "Max Reward: " + str(max_reward)
    print "Max State: " + str(max_state)
    
    game = Map(map)
    steps = 1000
    max_expereince = 10000
    # for i in range(steps):
    print action_selection
    i=0
    states = np.array([[0,0]])
    if settings['agent_name'] == "logistic":
        print "Creating Logistic agent"
        model = RLLogisticRegression(states, n_in=2, n_out=8)
    elif settings['agent_name'] == "NN":
        print "Creating NN agent"
        model = NeuralNet(states, n_in=2, n_out=8)
    elif settings['agent_name'] == "Deep":
        print "Creating Deep agent"
        model = RLNeuralNetwork(states, n_in=2, n_out=8)
    elif settings['agent_name'] == "Deep_DQ":
        print "Creating Deep agent"
        model = RLNeuralNetworkDQ(states, n_in=2, n_out=8)
    elif settings['agent_name'] == "Deep_NN":
        print "Creating Deep agent"
        model = RLDeepNet(states, n_in=2, n_out=8)
    else:
        print "Unrecognized model: " + str(settings['agent_name'])
    """ 
    if len(sys.argv) > 1:
        file_name=sys.argv[1]
        model = cPickle.load(open(file_name))
        """ 
    best_error=10000000.0
    X, Y, U, V, Q = get_policy_visual_data(model, max_state, game)
    game.init(U, V, Q)    
    experience = ExperienceMemory(2, 1, 5000)
    for round in range(rounds):
        game.reset()
        # reduces random action select probability
        p = (rounds - round) / float(rounds)
        t=0
        print "Random Action selection Pr(): " + str(p)
        while not game.reachedTarget():
            if (t > 20):
                game.reset()
                t=0
            state = game.getState()
            action = random.choice(action_selection)
            pa = model.predict([norm_state(state, max_state)])
            # print "policy action: " + str(pa) + " Q-values: " + str(model.q_values([norm_state(state, max_state)]))
            action = eGreedy(pa, action, epsilon * p)
            reward = game.act(action)
            resultState = game.getState()
            # tup = ExperienceTuple(state, [action], resultState, [reward])
            # Everything should be normalized to be between -1 and 1
            reward_ = (reward+(max_reward/2.0))/(max_reward*0.5)
            # reward_ = (reward)/(max_reward)
            # reward_ = (reward+max_reward)/(max_reward)
            experience.insert(norm_state(state, max_state), [action], norm_state(resultState, max_state), [reward_])
            # Update agent on screen
            # game.update()
            # X, Y, U, V, Q = get_policy_visual_data(model, max_state, game)
            # game.updatePolicy(U, V, Q)
            i += 1
            t += 1
            # print "Reward: " + str(reward_)
            # print "Reward for action " + str(tup._action) + " reward is " + str(tup._reward) + " State was " + str(tup._state)
            # print model.q_values([tup._state])
            if experience.samples() > batch_size:
                _states, _actions, _result_states, _rewards = experience.get_batch(batch_size)
                cost = model.train(_states, _actions, _rewards, _result_states)
                # print "Iteration: " + str(i) + " Cost: " + str(cost)
                
            if i % steps == 0:
                X, Y, U, V, Q = get_policy_visual_data(model, max_state, game)
                game.update()
                game.updatePolicy(U, V, Q)
                states, actions, result_states, rewards = experience.get_batch(32)
                error = model.bellman_error(states, actions, rewards, result_states)
                error = np.mean(np.fabs(error))
                print "Iteration: " + str(i) + " Cost: " + str(cost) + " Bellman Error: " + str(error)
    
        print ""
        # X,Y = np.mgrid[0:16,0:16]
        X, Y, U, V, Q = get_policy_visual_data(model, max_state, game)
        game.updatePolicy(U, V, Q)
        """
        states, actions, result_states, rewards = get_batch(experience, len(experience))
        error = model.bellman_error(states, actions, rewards, result_states)
        error = np.mean(np.fabs(error))
        print "Round: " + str(round) + " Iteration: " + str(i) + " Bellman Error: " + str(error) + " Expereince: " + str(len(experience))
        """
        # print model.q_values(states)[:5]
        # print experience[:10]
    
    # print "Experience: " + str(experience)
    print "Found target after " + str(i) + " actions"
    file_name="navigator_agent_"+str(settings['agent_name'])+".pkl"
    f = open(file_name, 'w')
    cPickle.dump(model, f)
    f.close()
    
