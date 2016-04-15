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
from RunGame import *


class Sampler(object):
    """
        A simple method to sample the space of actions.
    """    
    def __init__(self, game):
        self._x=[]
        self._samples=[]
        self._bestSample=([0],[-10000000])
        self._game=game
        

    def sample(self, game, state):
        self._samples = []
        self._bestSample=([0],[-10000000])
        xi = np.linspace(-2.0, 2.0, 100)
        for i in xi:
            y = game._reward(game._computeHeight(i+state[1]))
            # print i, y
            self._samples.append([[i],[y]])
            if y > self._bestSample[1][0]:
                self._bestSample[1][0] = y
                self._bestSample[0][0] = i

    def sampleModel(self, model, forwardDynamics, current_state, max_state, action_bounds):
        self._samples = []
        self._bestSample=([0],[-10000000])
        pa = model.predict([norm_state(current_state, max_state)])
        action = scale_action(pa, action_bounds)
        xi = np.linspace(-0.5+action[0], 0.5+action[0], 100)
        for i in xi:
            pa = [i]
            # prediction = scale_state(forwardDynamics.predict(state=norm_state(current_state, max_state), action=norm_action(pa, action_bounds)), max_state)
            # y = model.q_value([norm_state(prediction, max_state)])
            y = self._game._reward(self._game._computeHeight(i+current_state[1]))
            # print i, y
            self._samples.append([[i],[y]])
            if y > self._bestSample[1][0]:
                self._bestSample[1][0] = y
                self._bestSample[0][0] = i
                            
    def getBestSample(self): 
        return self._bestSample
    
    def predict(self, state):
        """
        Returns the best action
        """
        return np.random.rand(1,1)

    def q_value(self, state):
        """
        Returns the expected value of the state
        """
        return np.random.rand(1,1)[0]
    
def simpleSampling():
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
            print "Creating Game type: " + str(game_type)
            game = BallGame1DFuture()
        elif game_type == 'BallGame1D':
            print "Creating Game type: " + str(game_type)
            game = BallGame1D()
        else:
            print "Unrecognized game: " + str(game_type)
            sys.exit()
            
        game.enableRender()
        game._simulate=True
        # game._saveVideo=True
        game.setMovieName(str(settings['agent_name']) + "_on_" + str(game_type))
            
        action_bounds = np.array(settings['action_bounds'])
        action_length = len(action_bounds[0])
        data_folder = settings['data_folder']
        states = np.array([max_state])
        state_length = len(max_state)
        action_space_continuous=True
        
        # file_name=data_folder+"navigator_agent_"+str(settings['agent_name'])+".pkl"
        model = Sampler()
        
        file_name_dynamics=data_folder+"forward_dynamics_"+str(settings['agent_name'])+".pkl"
        forwardDynamicsModel = cPickle.load(open(file_name_dynamics))
        
        if action_space_continuous:
            # X, Y, U, V, Q = get_continuous_policy_visual_data(model, max_state, game)
            X, Y, U, V, Q = get_continuous_policy_visual_data1D(model, max_state, game)
        else:
            X, Y, U, V, Q = get_policy_visual_data(model, max_state, game)
        print "U: " + str(U)
        print "V: " + str(V)
        print "Q: " + str(Q)
        game.init(U, V, Q)
        game.reset()
        
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        num_actions = 20
        scaling = 1.0
        game._box.state[0][1] = 0.0
        
        actions = (np.random.rand(num_actions,1)-0.5) * 2.0 * scaling
        
        reward_sum=0
        for action_ in actions:
            # ballGame.resetTarget()
            game.resetTarget()
            game.resetHeight()
            game._box.state[0][1] = 0.0
            state = game.getState()
            print "State: " + str(state)
            model.sample(game, state)
            # reward = game.actContinuous(action_)
            # print "Action: " + str(action_)
            # print "Verify State: " + str(state) + " with " + str(scale_state(norm_state(state, max_state=max_state), max_state=max_state))
            if action_space_continuous:
                # X, Y, U, V, Q = get_continuous_policy_visual_data(model, max_state, game)
                X, Y, U, V, Q = get_continuous_policy_visual_data1D(model, max_state, game)
            else:
                X, Y, U, V, Q = get_policy_visual_data(model, max_state, game)
            game.updatePolicy(U, V, Q)
            # pa = model.predict([norm_state(state, max_state)])
            if action_space_continuous:
                # action = scale_action(pa, action_bounds)
                action = model.getBestSample()[:1]
                print "Action: " + str(action)
                # prediction = scale_state(forwardDynamicsModel.predict(state=norm_state(state, max_state), action=norm_action(action, action_bounds)), max_state)
                # print "Next State Prediction: " + str(prediction)
                # predicted_height = game._computeHeight(prediction[1]) # This is dependent on the network shape
                # game.setPrediction([2,predicted_height])
                # print "Next Height Prediction: " + str(predicted_height)
                reward = game.actContinuous(action)
                # print "Height difference: " + str(math.fabs(predicted_height - game._max_y))
            elif not action_space_continuous:
                # print "Action: " + str(pa)
                reward = game.act(action)
            reward_sum+=reward
                
            # print "Reward: " + str(reward)
            
        print "Average reward: " + str(reward_sum/num_actions)

    #except Exception, e:
    #    print "Error: " + str(e)
    #    raise e

def modelSampling():
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
            print "Creating Game type: " + str(game_type)
            game = BallGame1DFuture()
        elif game_type == 'BallGame1D':
            print "Creating Game type: " + str(game_type)
            game = BallGame1D()
        else:
            print "Unrecognized game: " + str(game_type)
            sys.exit()
            
        game.enableRender()
        game._simulate=True
        # game._saveVideo=True
        game.setMovieName(str(settings['agent_name']) + "_on_" + str(game_type))
            
        action_bounds = np.array(settings['action_bounds'])
        action_length = len(action_bounds[0])
        data_folder = settings['data_folder']
        states = np.array([max_state])
        state_length = len(max_state)
        action_space_continuous=True
        
        sampler = Sampler(game)
        
        file_name=data_folder+"navigator_agent_"+str(settings['agent_name'])+".pkl"
        model = cPickle.load(open(file_name))
        
        file_name_dynamics=data_folder+"forward_dynamics_"+str(settings['agent_name'])+".pkl"
        forwardDynamicsModel = cPickle.load(open(file_name_dynamics))
                
        if action_space_continuous:
            # X, Y, U, V, Q = get_continuous_policy_visual_data(model, max_state, game)
            X, Y, U, V, Q = get_continuous_policy_visual_data1D(model, max_state, game)
        else:
            X, Y, U, V, Q = get_policy_visual_data(model, max_state, game)
        # print "U: " + str(U)
        # print "V: " + str(V)
        # print "Q: " + str(Q)
        game.init(U, V, Q)
        game.reset()
        
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        num_actions = 20
        scaling = 1.0
        game._box.state[0][1] = 0.0
        
        actions = (np.random.rand(num_actions,1)-0.5) * 2.0 * scaling
        
        reward_sum=0
        for action_ in actions:
            # ballGame.resetTarget()
            game.resetTarget()
            game.resetHeight()
            game._box.state[0][1] = 0.0
            state = game.getState()
            print "State: " + str(state)
            sampler.sampleModel(model=model, forwardDynamics=forwardDynamicsModel, current_state=state, max_state=max_state, 
                                action_bounds=action_bounds)
            # reward = game.actContinuous(action_)
            # print "Action: " + str(action_)
            # print "Verify State: " + str(state) + " with " + str(scale_state(norm_state(state, max_state=max_state), max_state=max_state))
            if action_space_continuous:
                # X, Y, U, V, Q = get_continuous_policy_visual_data(model, max_state, game)
                X, Y, U, V, Q = get_continuous_policy_visual_data1D(model, max_state, game)
            else:
                X, Y, U, V, Q = get_policy_visual_data(model, max_state, game)
            game.updatePolicy(U, V, Q)
            # pa = model.predict([norm_state(state, max_state)])
            if action_space_continuous:
                # action = scale_action(pa, action_bounds)
                action = sampler.getBestSample()[:1]
                print "Action: " + str(action)
                # prediction = scale_state(forwardDynamicsModel.predict(state=norm_state(state, max_state), action=norm_action(action, action_bounds)), max_state)
                # print "Next State Prediction: " + str(prediction)
                # predicted_height = game._computeHeight(prediction[1]) # This is dependent on the network shape
                # game.setPrediction([2,predicted_height])
                # print "Next Height Prediction: " + str(predicted_height)
                reward = game.actContinuous(action)
                # print "Height difference: " + str(math.fabs(predicted_height - game._max_y))
            elif not action_space_continuous:
                # print "Action: " + str(pa)
                reward = game.act(action)
            reward_sum+=reward
                
            # print "Reward: " + str(reward)
            
        print "Average reward: " + str(reward_sum/num_actions)

    #except Exception, e:
    #    print "Error: " + str(e)
    #    raise e
    
if __name__ == "__main__":
    
    modelSampling()
    