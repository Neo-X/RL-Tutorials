from MapGame import Map
import random
import numpy as np
import math
import cPickle

from RLLogisticRegression import RLLogisticRegression
from NeuralNet import NeuralNet 
from RLNeuralNetwork import RLNeuralNetwork 

def loadMap():
    dataset="map.json"
    import json
    dataFile = open(dataset)
    s = dataFile.read()
    data = json.loads(s)
    dataFile.close()
    return data["map"]

class ExperienceTuple(object):

    def __init__(self, state, action, resultState, reward):
        self._state = state
        self._action = action
        self._resultState = resultState
        self._reward = reward
    
    def __str__(self): 
        return self.__repr__()
    
    def __repr__(self): 
        return "{" + str(self._state) + ", " + str(self._action) + ", " + str(self._resultState) + ", " + str(self._reward) + "}"




def get_datas(experiences):
    
    
    state = []
    action = []
    resultState = []
    reward = []
    
    for i in range(len(experiences)):
        state.append(experiences[i]._state)
        action.append(experiences[i]._action)
        resultState.append(experiences[i]._resultState)
        reward.append(experiences[i]._reward)
    # print c
    # print experience[indices]
    state = np.array(state)
    action = np.array(action)
    resultState = np.array(resultState)
    reward = np.array(reward)
     
    return (state, action, resultState, reward)

def get_batch(experience, batch_size=32):
    """
    len(experience > batch_size
    """
    # indices = list(nprnd.randint(low=0, high=len(experience), size=batch_size))
    indices = (random.sample(range(0, len(experience)), batch_size))
    # print indices
    c = [ experience[i] for i in indices]
    
    return get_datas(c)
    
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
    
def get_policy_visual_data(model_, max_state, game):
    X,Y = np.mgrid[0:16,0:16]
    U = []
    V = []
    for i in range(16):
        t_u = []
        t_v = []
        for j in range(16):
            state = np.array([X[i][j],Y[i][j]])
            pa = model_.predict([norm_state(state,max_state)])[0]
            move = game.move(pa)
            t_u.append(move[0])
            t_v.append(move[1])
        U.append(t_u)
        V.append(t_v)
    
    U = np.array(U)*2.0
    V = np.array(V)*2.0
    return (U,V)

def norm_state(state, max_state):
    return (state-max_state)/max_state
    
    
if __name__ == "__main__":
    
    # make a color map of fixed colors
    
    batch_size=32
    rounds = 200    
    epsilon = 0.8
    map = loadMap()
    # Normalization constants for data
    max_reward = math.sqrt(16**2 * 2) + 5
    # max_reward = 1.0
    max_state = 8.0
    
    num_actions=8
    action_selection = range(num_actions)
    
    print "Max Reward: " + str(max_reward)
    print "Max State: " + str(max_state)
    
    game = Map(map)
    steps = 100
    max_expereince = 10000
    # for i in range(steps):
    print action_selection
    i=0
    states = np.array([[0,0]])
    # model = RLLogisticRegression(states, n_in=2, n_out=8)
    # model = NeuralNet(states, n_in=2, n_out=8)
    model = RLNeuralNetwork(states, n_in=2, n_out=8)
    best_error=10000000.0
    U,V = get_policy_visual_data(model, max_state, game)
    game.init(U,V)    
    experience = []
    for round in range(rounds):
        game.reset()
        # reduces random action select probability
        p = (rounds - round) / float(rounds)
        print "Random Action selection p: " + str(p)
        while not game.reachedTarget():
            state = game.getState()
            action = random.choice(action_selection)
            pa = model.predict([norm_state(state, max_state)])[0]
            # print "policy action: " + str(pa) + " Q-values: " + str(model._q_values([(state-max_state)/max_state]))
            action = eGreedy(pa, action, epsilon * p)
            reward = game.act(action)
            resultState = game.getState()
            # tup = ExperienceTuple(state, [action], resultState, [reward])
            # Everything should be normalized to be between -1 and 1
            tup = ExperienceTuple(norm_state(state, max_state), [action], norm_state(resultState, max_state), [reward/max_reward])
            experience.append(tup)
            # Update agent on screen
            # game.update()
            # U,V = get_policy_visual_data(model, max_state, game)
            # game.updatePolicy(U, V)
            i +=1
            # print "Reward for action " + str(action) + " is " + str(reward) + " State was " + str(state)
            if len(experience) > max_expereince:
                experience.pop(0)
            if len(experience) > batch_size:
                _states, _actions, _result_states, _rewards = get_batch(experience, batch_size)
                cost = model.train(_states, _rewards, _result_states)
                # print "Iteration: " + str(i) + " Cost: " + str(cost)
                
            if i % 100 == 0:
                U,V = get_policy_visual_data(model, max_state, game)
                game.update()
                game.updatePolicy(U, V)
                states, actions, result_states, rewards = get_batch(experience, len(experience))
                error = model.bellman_error(states, rewards, result_states)
                error = np.mean(np.fabs(error))
                print "Iteration: " + str(i) + " Cost: " + str(cost) + " Bellman Error: " + str(error)
    
        print ""
        # X,Y = np.mgrid[0:16,0:16]
        U,V = get_policy_visual_data(model, max_state, game)
        game.updatePolicy(U, V)
        states, actions, result_states, rewards = get_batch(experience, len(experience))
        error = model.bellman_error(states, rewards, result_states)
        error = np.mean(np.fabs(error))
        print "Round: " + str(round) + " Iteration: " + str(i) + " Bellman Error: " + str(error) + " Expereince: " + str(len(experience))
        print model.q_values(states)[:10]
        # print experience[:10]
    
    # print "Experience: " + str(experience)
    print "Found target after " + str(i) + " actions"
    file_name="navigator_agent.pkl"
    f = open(file_name, 'w')
    cPickle.dump(model, f)
    f.close()
    
