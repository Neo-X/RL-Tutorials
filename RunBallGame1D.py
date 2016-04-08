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
        batch_size=32
        rounds = 1000
        max_training_steps=2000000
        
        epsilon = 0.45 # It is important to have some space between these values especially now that the experience buffer starts loaded with random actions
        omega = 0.8
        map = loadMap()
        # Normalization constants for data
        max_reward = settings['max_reward']
        # max_reward = 1.0
        max_state = settings['max_state']
        
        num_actions=8
        action_selection = range(num_actions)
        
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
            
        game._simulate=False
        steps = 500
        max_expereince = 20000
        # for i in range(steps):
        print action_selection
        i=0
        action_bounds = np.array(settings['action_bounds'])
        action_length = len(action_bounds[0])
        data_folder = settings['data_folder']
        states = np.array([max_state])
        state_length = len(max_state)
        action_space_continuous=False
        if settings['agent_name'] == "logistic":
            print "Creating Logistic agent"
            model = RLLogisticRegression(states, n_in=state_length, n_out=8)
        elif settings['agent_name'] == "NN":
            print "Creating NN agent"
            model = NeuralNet(states, n_in=state_length, n_out=8)
        elif settings['agent_name'] == "Deep":
            print "Creating Deep agent"
            model = RLNeuralNetwork(states, n_in=state_length, n_out=8)
        elif settings['agent_name'] == "Deep_DQ":
            print "Creating Deep agent"
            model = RLNeuralNetworkDQ(states, n_in=state_length, n_out=8)
        elif settings['agent_name'] == "Deep_NN":
            print "Creating Deep agent"
            model = RLDeepNet(states, n_in=state_length, n_out=8)
            max_training_steps = settings['max_training_steps']
            epsilon = settings['epsilon']
        elif settings['agent_name'] == "Deep_CACLA":
            print "Creating " + str(settings['agent_name']) + " agent"
            model = DeepCACLA(n_in=state_length, n_out=action_length)
            action_space_continuous=True
        elif settings['agent_name'] == "Deep_DPG":
            print "Creating " + str(settings['agent_name']) + " agent"
            model = DeepDPG(n_in=state_length, n_out= action_length)
            action_space_continuous=True
        else:
            print "Unrecognized model: " + str(settings['agent_name'])
            sys.exit()
        """ 
        if len(sys.argv) > 1:
            file_name=sys.argv[1]
            model = cPickle.load(open(file_name))
        """
        
        forwardDynamicsModel = ForwardDynamicsNetwork(state_length=state_length,action_length=action_length)
        values = []
        discounted_values = []
        bellman_error = []
        reward_over_epoc = []
        
        trainData = {}
        trainData["mean_reward"]=[]
        trainData["std_reward"]=[]
        trainData["mean_bellman_error"]=[]
        trainData["std_bellman_error"]=[]
        trainData["mean_discount_error"]=[]
        trainData["std_discount_error"]=[]
        trainData["mean_forward_dynamics_loss"]=[]
        trainData["std_forward_dynamics_loss"]=[]
         
        best_error=10000000.0
        if action_space_continuous:
            # X, Y, U, V, Q = get_continuous_policy_visual_data(model, max_state, game)
            X, Y, U, V, Q = get_continuous_policy_visual_data1D(model, max_state, game)
        else:
            X, Y, U, V, Q = get_policy_visual_data(model, max_state, game)
        game.init(U, V, Q)
        
        rlv = RLVisualize(title=str(settings['agent_name'] + " on " + str(game_type)))
        rlv.setInteractive()
        rlv.init()
        nlv = NNVisualize(title=str("Forward Dynamics Model") + " on " + str(game_type))
        nlv.setInteractive()
        nlv.init()
        
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        if action_space_continuous:
            experience = ExperienceMemory(state_length=state_length, action_length=action_length, memory_length=max_expereince, continuous_actions=action_space_continuous)
            # experience = collectExperienceActionsContinuous(experience, action_bounds)
        else: 
            experience = ExperienceMemory(state_length=state_length, action_length=action_length, memory_length=max_expereince)
        bellman_errors = []
        reward_over_epocs = []
        values = []
        step=0
        while step < max_training_steps:
            game.reset()
            # reduces random action select probability
            p = (max_training_steps - step) / float(max_training_steps)
            t=0
            print "Random Action selection Pr(): " + str(p)
            discounted_values = []
            bellman_errors = []
            reward_over_epocs = []
            dynamicsLosses = []
            values = []
            states = [] 
            actions = []
            rewards = []
            result_states = []
            discounted_sum = 0;
            reward_sum=0
            state_num=0
            state_ = game.getState()
            q_value = model.q_value([norm_state(state_, max_state)])
            action_ = model.predict([norm_state(state_, max_state)])
            print "q_values: " + str(q_value) + " Action: " + str(action_) + " State: " + str([norm_state(state_, max_state)])
            original_val = q_value
            values.append(original_val)
            t=0
            while t < 642:
                step+=1
                if (((t % 32) == 0) and (t > 0) ):
                    game.reset()
                    # print "Reward o epochs: " + str(reward_over_epocs)
                    reward_over_epocs.append(reward_sum)
                    discounted_values.append(discounted_sum)
                    # print "Actions: " + str(actions)
                    # states, actions, result_states, rewards = experience.get_batch(batch_size)
                    # print "States: " + str(states)
                    error = model.bellman_error(np.array(states), np.array(actions), 
                                np.array(rewards), np.array(result_states))
                    dynamicsLoss = forwardDynamicsModel.bellman_error(np.array(states), np.array(actions), np.array(result_states))
                    # states, actions, result_states, rewards = experience.get_batch(64)
                    # error = model.bellman_error(states, actions, rewards, result_states)
                    # print "Error: " + str(error)
                    error = np.mean(np.fabs(error))
                    bellman_errors.append(error)
                    dynamicsLoss = np.mean(np.fabs(dynamicsLoss))
                    dynamicsLosses.append(dynamicsLoss)
                    
                    discounted_sum = 0;
                    reward_sum=0
                    state_num=0
                    
                    states = [] 
                    actions = []
                    rewards = []
                    result_states = []
                    error = None
                    
                    
                state = game.getState()
                pa = model.predict([norm_state(state, max_state)])
                
                if action_space_continuous:
                    pa = scale_action(pa, action_bounds)
                    action = randomExporation(0.12, pa)
                    randomAction = randomUniformExporation(action_bounds) # Completely random action
                    # print "policy action: " + str(pa) + " Q-values: " + str(model.q_values([norm_state(state, max_state)]))
                    action = eOmegaGreedy(pa, action, randomAction, epsilon * p, omega * p)
                    action = clampAction(action, action_bounds)
                    reward = game.actContinuous(action)
                elif not action_space_continuous:
                    action = random.choice(action_selection)
                    action = eGreedy(pa, action, epsilon * p)
                    reward = game.act(action)
                    
                if reward is None:
                    # something bad happened
                    reward = max_reward
                resultState = game.getState()
                # tup = ExperienceTuple(state, [action], resultState, [reward])
                # Everything should be normalized to be between -1 and 1
                reward_ = reward
                # print "Reward: " + str(reward_)
                # reward_ = (reward)/(max_reward)
                # reward_ = (reward+max_reward)/(max_reward)
                experience.insert(norm_state(state, max_state), norm_action(action, action_bounds), norm_state(resultState, max_state), [reward_])
                game.resetTarget()
                # Update agent on screen
                # game.update()
                # X, Y, U, V, Q = get_policy_visual_data(model, max_state, game)
                # game.updatePolicy(U, V, Q)
                # print "Reward: " + str(reward_)
                # print "Reward for action " + str(tup._action) + " reward is " + str(tup._reward) + " State was " + str(tup._state)
                # print model.q_values([tup._state])
                actions.append(norm_action(action, action_bounds))
                result_states.append(norm_state(resultState, max_state))
                rewards.append([reward_])
                states.append(norm_state(state, max_state))
                reward_sum+=reward_
                discounted_sum += (math.pow(0.8,t) * reward)
                if experience.samples() > batch_size:
                    _states, _actions, _result_states, _rewards = experience.get_batch(batch_size)
                    # print _states, _rewards
                    cost = model.train(_states, _actions, _rewards, _result_states)
                    dynamicsLoss = forwardDynamicsModel.train(_states, _actions, _result_states)
                    # print "Iteration: " + str(i) + " Cost: " + str(cost)
                i += 1
                t += 1
                    
            if action_space_continuous:
                X, Y, U, V, Q = get_continuous_policy_visual_data1D(model, max_state, game)
            else:
                X, Y, U, V, Q = get_policy_visual_data(model, max_state, game)
            game.update()
            game.updatePolicy(U, V, Q)
            states_, actions_, result_states_, rewards_ = experience.get_batch(batch_size)
            error = model.bellman_error(states_, actions_, rewards_, result_states_)
            error = np.mean(np.fabs(error))
            print "Iteration: " + str(i) + "RL Loss: " + str(cost) + " Bellman Error: " + str(error) + " dynamicsLoss: " + str(dynamicsLoss)
            # print "Reward over epochs: " + str(reward_over_epocs)
            mean_reward = np.mean(reward_over_epocs)
            std_reward = np.std(reward_over_epocs)
            mean_bellman_error = np.mean(bellman_errors)
            std_bellman_error = np.std(bellman_errors)
            mean_discount_error = np.mean(np.array(discounted_values) - np.array(values))
            std_discount_error = np.std(np.array(discounted_values) - np.array(values))
            mean_dynamicsLosses = np.mean(dynamicsLosses)
            std_dynamicsLosses = np.std(dynamicsLosses)
            
            trainData["mean_reward"].append(mean_reward)
            # print "Mean Rewards: " + str(trainData["mean_reward"])
            trainData["std_reward"].append(std_reward)
            trainData["mean_bellman_error"].append(mean_bellman_error)
            # print "beelman error: " + str(trainData["mean_bellman_error"])
            trainData["std_bellman_error"].append(std_bellman_error)
            trainData["mean_discount_error"].append(mean_discount_error)
            trainData["std_discount_error"].append(std_discount_error)
            trainData["mean_forward_dynamics_loss"].append(mean_dynamicsLosses)
            trainData["std_forward_dynamics_loss"].append(mean_dynamicsLosses)
            
            rlv.updateBellmanError(np.array(trainData["mean_bellman_error"]), np.array(trainData["std_bellman_error"]))
            rlv.updateReward(np.array(trainData["mean_reward"]), np.array(trainData["std_reward"]))
            rlv.updateDiscountError(np.fabs(trainData["mean_discount_error"]), np.array(trainData["std_discount_error"]))
            rlv.redraw()
            nlv.updateLoss(np.array(trainData["mean_forward_dynamics_loss"]), np.array(trainData["std_forward_dynamics_loss"]))
            nlv.redraw()
                    
            
            reward_over_epocs.append(reward_sum)
            discounted_values.append(discounted_sum)
            
            # error = model.bellman_error(np.array(states), np.array(actions), 
            #          np.array(rewards), np.array(result_states))
            # error = np.mean(np.fabs(error))
            # bellman_errors.append(0)
            
            states = [] 
            actions = []
            rewards = []
            result_states = []
            
            rlv.redraw()
            rlv.setInteractiveOff()
            rlv.saveVisual(data_folder+"trainingGraph")
            rlv.setInteractive()
            nlv.setInteractiveOff()
            nlv.saveVisual(data_folder+"trainingGraphNN")
            nlv.setInteractive()
                
            print ""
            # X,Y = np.mgrid[0:16,0:16]
            if action_space_continuous:
                X, Y, U, V, Q = get_continuous_policy_visual_data1D(model, max_state, game)
            else:
                X, Y, U, V, Q = get_policy_visual_data(model, max_state, game)
            game.updatePolicy(U, V, Q)
            game.saveVisual(data_folder+"gameState")
            """
            states, actions, result_states, rewards = get_batch(experience, len(experience))
            error = model.bellman_error(states, actions, rewards, result_states)
            error = np.mean(np.fabs(error))
            print "Round: " + str(round) + " Iteration: " + str(i) + " Bellman Error: " + str(error) + " Expereince: " + str(len(experience))
            """
            file_name=data_folder+"navigator_agent_"+str(settings['agent_name'])+".pkl"
            f = open(file_name, 'w')
            cPickle.dump(model, f)
            f.close()
            
            file_name_dynamics=data_folder+"forward_dynamics_"+str(settings['agent_name'])+".pkl"
            f = open(file_name_dynamics, 'w')
            cPickle.dump(forwardDynamicsModel, f)
            f.close()
            # print model.q_values(states)[:5]
            # print experience[:10]
        
        # print "Experience: " + str(experience)
        print "Found target after " + str(i) + " actions"
        file_name=data_folder+"navigator_agent_"+str(settings['agent_name'])+".pkl"
        f = open(file_name, 'w')
        cPickle.dump(model, f)
        f.close()
        
        file_name_dynamics=data_folder+"forward_dynamics_"+str(settings['agent_name'])+".pkl"
        f = open(file_name_dynamics, 'w')
        cPickle.dump(forwardDynamicsModel, f)
        f.close()
    #except Exception, e:
    #    print "Error: " + str(e)
    #    raise e
