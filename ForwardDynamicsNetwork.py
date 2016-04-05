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

# For debugging
# theano.config.mode='FAST_COMPILE'

class ForwardDynamicsNetwork(object):
    
    def __init__(self, state_length, action_length):

        batch_size=32
        # data types for model
        State = T.dmatrix("State")
        State.tag.test_value = np.random.rand(batch_size,state_length)
        ResultState = T.dmatrix("ResultState")
        ResultState.tag.test_value = np.random.rand(batch_size,state_length)
        Action = T.dmatrix("Action")
        Action.tag.test_value = np.random.rand(batch_size, action_length)
        # create a small convolutional neural network
        inputLayerState = lasagne.layers.InputLayer((None, state_length), State)
        inputLayerAction = lasagne.layers.InputLayer((None, action_length), Action)
        concatLayer = lasagne.layers.ConcatLayer([inputLayerState, inputLayerAction])
        l_hid2ActA = lasagne.layers.DenseLayer(
                concatLayer, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid3ActA = lasagne.layers.DenseLayer(
                l_hid2ActA, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid4ActA = lasagne.layers.DenseLayer(
                l_hid3ActA, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
    
        self._l_out = lasagne.layers.DenseLayer(
                l_hid4ActA, num_units=state_length,
                nonlinearity=lasagne.nonlinearities.linear)
                # print "Initial W " + str(self._w_o.get_value()) 
        
        self._learning_rate = 0.001
        self._discount_factor= 0.8
        self._rho = 0.95
        self._rms_epsilon = 0.001
        
        self._weight_update_steps=5
        self._updates=0
        
        self._states_shared = theano.shared(
            np.zeros((batch_size, state_length),
                     dtype=theano.config.floatX))

        self._next_states_shared = theano.shared(
            np.zeros((batch_size, state_length),
                     dtype=theano.config.floatX))

        self._actions_shared = theano.shared(
            np.zeros((batch_size, action_length), dtype=theano.config.floatX),
            )
        
        inputs_ = {
            State: self._states_shared,
            Action: self._q_valsActA,
        }
        self._forward = lasagne.layers.get_output(self._l_out, inputs_)
        
        # self._target = (Reward + self._discount_factor * self._q_valsB)
        self._diff = self._next_states_shared - self._forward
        self._loss = 0.5 * self._diff ** 2 + (1e-4 * lasagne.regularization.regularize_network_params(
                self._l_outA, lasagne.regularization.l2))
        self._loss = T.mean(self._loss)
        
        self._params = lasagne.layers.helper.get_all_params(self._l_out)
        self._actionParams = lasagne.layers.helper.get_all_params(self._l_outActA)
        self._givens_ = {
            State: self._states_shared,
            ResultState: self._next_states_shared,
            Action: self._actions_shared,
        }
        
        # SGD update
        self._updates_ = lasagne.updates.rmsprop(self._loss, self._params, self._learning_rate, self._rho,
                                            self._rms_epsilon)
        # TD update
        # minimize Value function error
        #self._updates_ = lasagne.updates.rmsprop(T.mean(self._q_func) + (1e-4 * lasagne.regularization.regularize_network_params(
        #self._l_outA, lasagne.regularization.l2)), self._params, 
        #            self._learning_rate * -T.mean(self._diff), self._rho, self._rms_epsilon)
        
        
        # actDiff1 = (Action - self._q_valsActB) #TODO is this correct?
        # actDiff = (actDiff1 - (Action - self._q_valsActA))
        # actDiff = ((Action - self._q_valsActB2)) # Target network does not work well here?
        #self._actDiff = ((Action - self._q_valsActA)) # Target network does not work well here?
        #self._actLoss = 0.5 * self._actDiff ** 2 + (1e-4 * lasagne.regularization.regularize_network_params( self._l_outActA, lasagne.regularization.l2))
        #self._actLoss = T.mean(self._actLoss)
        
        
        
        
        self._train = theano.function([], [self._loss, self._q_func], updates=self._updates_, givens=self._givens_)
        self._forwardDynamics = theano.function([], self._forward,
                                       givens={State: self._states_shared, Action: self._actions_shared})
        inputs_ = [
                   State, 
                   ResultState,
                   Action
                   ]
        self._bellman_error = theano.function(inputs=inputs_, outputs=self._diff, allow_input_downcast=True)
        # self._diffs = theano.function(input=[State])
        
    def train(self, states, actions, result_states):
        self._states_shared.set_value(states)
        self._next_states_shared.set_value(result_states)
        self._actions_shared.set_value(actions)
        # print "Performing Critic trainning update"
        if (( self._updates % self._weight_update_steps) == 0):
            self.updateTargetModel()
        self._updates += 1
        # all_paramsActA = lasagne.layers.helper.get_all_param_values(self._l_outActA)
        loss, _ = self._train()
        # This undoes the Actor parameter updates as a result of the Critic update.
        #if all_paramsActA == self._l_outActA:
        #    print "Parameters the same:"
        # lasagne.layers.helper.set_all_param_values(self._l_outActA, all_paramsActA)
        # self._trainOneActions(states, actions, rewards, result_states)
        # diff_ = self._bellman_error(states, rewards, result_states)
        # print "Diff"
        # print diff_
        return loss
    
    def predict(self, state, action):
        # states = np.zeros((self._batch_size, self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        self._states_shared.set_value(state)
        self._action_shared.set_value(action)
        action_ = self._forwardDynamics()[0]
        return action_

    def bellman_error(self, state, action, reward, result_state):
        # return self._bellman_error(state, reward, result_state)
        return self._bellman_error(state, result_state, action)
