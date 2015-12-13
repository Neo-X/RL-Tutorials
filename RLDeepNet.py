import theano
from theano import tensor as T
import numpy as np
import lasagne

def sgd(cost, params, lr=0.05):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p + (-g * lr)])
    return updates

def rlTDSGD(cost, delta, params, lr=0.05):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p + (lr * delta * g)])
    return updates


class RLDeepNet(object):
    
    def __init__(self, input, n_in, n_out):

        batch_size=32
        state_length=2
        # data types for model
        State = T.dmatrix("State")
        State.tag.test_value = np.random.rand(batch_size,state_length)
        ResultState = T.dmatrix("ResultState")
        ResultState.tag.test_value = np.random.rand(batch_size,state_length)
        Reward = T.col("Reward")
        Reward.tag.test_value = np.random.rand(batch_size,1)
        Action = T.icol("Action")
        Action.tag.test_value = np.zeros((batch_size,1),dtype=np.dtype('int32'))
        # create a small convolutional neural network
        inputLayerA = lasagne.layers.InputLayer((None, state_length), State)

        l_hid1A = lasagne.layers.DenseLayer(
                inputLayerA, num_units=64,
                nonlinearity=lasagne.nonlinearities.rectify)
        
        l_hid2A = lasagne.layers.DenseLayer(
                l_hid1A, num_units=32,
                nonlinearity=lasagne.nonlinearities.rectify)
    
        self._l_outA = lasagne.layers.DenseLayer(
                l_hid2A, num_units=8,
                nonlinearity=lasagne.nonlinearities.linear)
        # self._b_o = init_b_weights((n_out,))
        
        # self.updateTargetModel()
        inputLayerB = lasagne.layers.InputLayer((None, state_length), State)

        l_hid1B = lasagne.layers.DenseLayer(
                inputLayerB, num_units=64,
                nonlinearity=lasagne.nonlinearities.rectify)
        
        l_hid2B = lasagne.layers.DenseLayer(
                l_hid1B, num_units=32,
                nonlinearity=lasagne.nonlinearities.rectify)
    
        self._l_outB = lasagne.layers.DenseLayer(
                l_hid2B, num_units=8,
                nonlinearity=lasagne.nonlinearities.linear)

        
        # print "Initial W " + str(self._w_o.get_value()) 
        
        self._learning_rate = 0.00025
        self._discount_factor= 0.99
        self._rho = 0.95
        self._rms_epsilon = 0.01
        
        self._weight_update_steps=5000
        self._updates=0
        
        self._states_shared = theano.shared(
            np.zeros((batch_size, state_length),
                     dtype=theano.config.floatX))

        self._next_states_shared = theano.shared(
            np.zeros((batch_size, state_length),
                     dtype=theano.config.floatX))

        self._rewards_shared = theano.shared(
            np.zeros((batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))

        self._actions_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))
        
        self._q_valsA = lasagne.layers.get_output(self._l_outA, State)
        self._q_valsB = lasagne.layers.get_output(self._l_outB, ResultState)
        
        self._q_func = self._q_valsA[T.arange(batch_size), Action.reshape((-1,))].reshape((-1, 1))
        
        target = (Reward +
                #(T.ones_like(terminals) - terminals) *
                  self._discount_factor * T.max(self._q_valsB, axis=1, keepdims=True))
        diff = target - self._q_valsA[T.arange(batch_size),
                               Action.reshape((-1,))].reshape((-1, 1))
                               
        loss = 0.5 * diff ** 2
        loss = T.mean(loss)
        
        params = lasagne.layers.helper.get_all_params(self._l_outA)
        
        givens = {
            State: self._states_shared,
            ResultState: self._next_states_shared,
            Reward: self._rewards_shared,
            Action: self._actions_shared,
        }
        
        # SGD update
        updates = lasagne.updates.rmsprop(loss, params, self._learning_rate, self._rho,
                                             self._rms_epsilon)
        # TD update
        #updates = lasagne.updates.rmsprop(T.mean(self._q_func), params, self._learning_rate * -T.mean(diff), self._rho,
        #                                      self._rms_epsilon)
        
        
        
        self._train = theano.function([], [loss, self._q_valsA], updates=updates,
                                      givens=givens)
        self._q_vals = theano.function([], self._q_valsA,
                                       givens={State: self._states_shared})
        
        self._bellman_error = theano.function(inputs=[State, Action, Reward, ResultState], outputs=diff, allow_input_downcast=True)
        
    def updateTargetModel(self):
        print "Updating target Model"
        all_params = lasagne.layers.helper.get_all_param_values(self._l_outA)
        lasagne.layers.helper.set_all_param_values(self._l_outB, all_params) 
    
    def train(self, states, actions, rewards, result_states):
        self._states_shared.set_value(states)
        self._next_states_shared.set_value(result_states)
        self._actions_shared.set_value(actions)
        self._rewards_shared.set_value(rewards)

        if (( self._updates % self._weight_update_steps) == 0):
            self.updateTargetModel()
        self._updates += 1
        loss, _ = self._train()
        return np.sqrt(loss)
    
    def predict(self, state):
        q_vals = self.q_values(state)
        return np.argmax(q_vals)
    def q_values(self, state):
        # states = np.zeros((self._batch_size, self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        self._states_shared.set_value(state)
        return self._q_vals()[0]
    def bellman_error(self, state, action, reward, result_state):
        return self._bellman_error(state, action, reward, result_state)
