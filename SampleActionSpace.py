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
from BallGame1DChoice import BallGame1DChoice

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
        # action, value
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

    def sampleModel(self, model, forwardDynamics, current_state, state_bounds, action_bounds):
        self._samples = []
        self._bestSample=([0],[-10000000])
        pa = model.predict([norm_state(current_state, state_bounds)])
        action = scale_action(pa, action_bounds)
        xi = np.linspace(-0.5+action[0], 0.5+action[0], 100)
        for i in xi:
            pa = [i]
            # prediction = scale_state(forwardDynamics.predict(state=norm_state(current_state, state_bounds), action=norm_action(pa, action_bounds)), state_bounds)
            # y = model.q_value([norm_state(prediction, state_bounds)])
            y = self._game._reward(self._game._computeHeight(i+current_state[1]))
            # print i, y
            self._samples.append([[i],[y]])
            if y > self._bestSample[1][0]:
                self._bestSample[1][0] = y
                self._bestSample[0][0] = i
                            
    def getBestSample(self): 
        return self._bestSample
    
    def setBestSample(self, samp): 
        self._bestSample = samp
    
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
        state_bounds = np.array(settings['state_bounds'])
        state_length = len(state_bounds[0])
        
        print "Max Reward: " + str(max_reward)
        print "State Bounds: " + str(state_bounds)
        
        # game = Map(map)
        game = None
        game_type = settings['game_type']
        if game_type == 'BallGame1DFuture':
            print "Creating Game type: " + str(game_type)
            game = BallGame1DFuture()
        elif game_type == 'BallGame1D':
            print "Creating Game type: " + str(game_type)
            game = BallGame1D()
        elif game_type == 'BallGame1DChoice':
            print "Creating Game type: " + str(game_type)
            game = BallGame1DChoice()
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
        states = np.array([state_bounds])
        action_space_continuous=True
        
        # file_name=data_folder+"navigator_agent_"+str(settings['agent_name'])+".pkl"
        model = Sampler()
        
        file_name_dynamics=data_folder+"forward_dynamics_"+str(settings['agent_name'])+".pkl"
        forwardDynamicsModel = cPickle.load(open(file_name_dynamics))
        
        if action_space_continuous:
            # X, Y, U, V, Q = get_continuous_policy_visual_data(model, state_bounds, game)
            X, Y, U, V, Q = get_continuous_policy_visual_data1D(model, state_bounds, game)
        else:
            X, Y, U, V, Q = get_policy_visual_data(model, state_bounds, game)
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
            # print "Verify State: " + str(state) + " with " + str(scale_state(norm_state(state, state_bounds=state_bounds), state_bounds=state_bounds))
            if action_space_continuous:
                # X, Y, U, V, Q = get_continuous_policy_visual_data(model, state_bounds, game)
                X, Y, U, V, Q = get_continuous_policy_visual_data1D(model, state_bounds, game)
            else:
                X, Y, U, V, Q = get_policy_visual_data(model, state_bounds, game)
            game.updatePolicy(U, V, Q)
            # pa = model.predict([norm_state(state, state_bounds)])
            if action_space_continuous:
                # action = scale_action(pa, action_bounds)
                action = model.getBestSample()[:1]
                print "Action: " + str(action)
                # prediction = scale_state(forwardDynamicsModel.predict(state=norm_state(state, state_bounds), action=norm_action(action, action_bounds)), state_bounds)
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
        state_bounds = np.array(settings['state_bounds'])
        state_length = len(state_bounds[0])
        
        print "Max Reward: " + str(max_reward)
        print "State Bounds: " + str(state_bounds)
        game_has_choices=False
        
        # game = Map(map)
        game = None
        game_type = settings['game_type']
        if game_type == 'BallGame1DFuture':
            print "Creating Game type: " + str(game_type)
            game = BallGame1DFuture()
        elif game_type == 'BallGame1D':
            print "Creating Game type: " + str(game_type)
            game = BallGame1D()
        elif game_type == 'BallGame1DChoice':
            print "Creating Game type: " + str(game_type)
            game = BallGame1DChoice()
            game_has_choices=True
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
        states = np.array([state_bounds])
        action_space_continuous=True
        
        sampler = Sampler(game)
        
        file_name=data_folder+"navigator_agent_"+str(settings['agent_name'])+".pkl"
        model = cPickle.load(open(file_name))
        
        file_name_dynamics=data_folder+"forward_dynamics_"+str(settings['agent_name'])+".pkl"
        forwardDynamicsModel = cPickle.load(open(file_name_dynamics))
                
        if action_space_continuous:
            # X, Y, U, V, Q = get_continuous_policy_visual_data(model, state_bounds, game)
            X, Y, U, V, Q = get_continuous_policy_visual_data1D(model, state_bounds, game)
        else:
            X, Y, U, V, Q = get_policy_visual_data(model, state_bounds, game)
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
            if (game_has_choices):
                action_v=-1000000000
                for i in range(len(state)):
                    state_ = state[i]
                    sampler.sampleModel(model=model, forwardDynamics=forwardDynamicsModel, current_state=state_, state_bounds=state_bounds, 
                                    action_bounds=action_bounds)     
                    action_ = sampler.getBestSample()
                    if (action_[1][0] > action_v):
                        action_v = action_[1][0] 
                        action = action_
                        state__ = state_
                        game.setTargetChoice(i)
                sampler.setBestSample(action)
                state = state__
            else:
                sampler.sampleModel(model=model, forwardDynamics=forwardDynamicsModel, current_state=state, state_bounds=state_bounds, 
                                action_bounds=action_bounds)
            # reward = game.actContinuous(action_)
            # print "Action: " + str(action_)
            # print "Verify State: " + str(state) + " with " + str(scale_state(norm_state(state, state_bounds=state_bounds), state_bounds=state_bounds))
            if action_space_continuous:
                # X, Y, U, V, Q = get_continuous_policy_visual_data(model, state_bounds, game)
                X, Y, U, V, Q = get_continuous_policy_visual_data1D(model, state_bounds, game)
            else:
                X, Y, U, V, Q = get_policy_visual_data(model, state_bounds, game)
            game.updatePolicy(U, V, Q)
            # pa = model.predict([norm_state(state, state_bounds)])
            if action_space_continuous:
                # action = scale_action(pa, action_bounds)
                action = sampler.getBestSample()[0]
                print "Action: " + str(action)
                prediction = scale_state(forwardDynamicsModel.predict(state=norm_state(state, state_bounds), action=norm_action(action, action_bounds)), state_bounds)
                # print "Next State Prediction: " + str(prediction)
                predicted_height = game._computeHeight(prediction[1]) # This is dependent on the network shape
                game.setPrediction([2,predicted_height])
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
    