import theano
from theano import tensor as T
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

srng = RandomStreams()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.5))

def init_b_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.0 + 0.1))

def init_tanh(n_in, n_out):
    rng = np.random.RandomState(1234)
    return theano.shared(np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            ))

def sgd(cost, params, lr=0.05):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p + (-g * lr)])
    return updates

def rectify(X):
    # return X
    return T.maximum(X, 0.)

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def dropout(X, p=0.):
    p=0.0 # diabled dropout
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

class RLNeuralNetwork(object):
    
    def __init__(self, input, n_in, n_out):

        hidden_size=36
        self._w_h = init_weights((n_in, hidden_size))
        self._b_h = init_b_weights((hidden_size,))
        self._w_h2 = init_weights((hidden_size, hidden_size))
        self._b_h2 = init_b_weights((hidden_size,))
        self._w_o = init_tanh(hidden_size, n_out)
        self._b_o = init_b_weights((n_out,))
        
        self.updateTargetModel()
        self._w_h = init_weights((n_in, hidden_size))
        self._w_h2 = init_weights((hidden_size, hidden_size))
        self._w_o = init_tanh(hidden_size, n_out)

        
        # print "Initial W " + str(self._w_o.get_value()) 
        
        self._learning_rate = 0.00025
        self._discount_factor= 0.8
        
        self._weight_update_steps=5000
        self._updates=0
        
        
        State = T.fmatrix("State")
        ResultState = T.fmatrix("ResultState")
        Reward = T.col("Reward")
        Action = T.icol("Action")
        # Q_val = T.fmatrix()
        
        # model = T.nnet.sigmoid(T.dot(State, self._w) + self._b.reshape((1, -1)))
        # self._model = theano.function(inputs=[State], outputs=model, allow_input_downcast=True)
        py_x = self.model(State, self._w_h, self._b_h, self._w_h2, self._b_h2, self._w_o, self._b_o, 0.0, 0.0)
        y_pred = T.argmax(py_x, axis=1)
        # q_val = py_x
        # noisey_q_val = self.model(ResultState, self._w_h, self._b_h, self._w_h2, self._b_h2, self._w_o, self._b_o, 0.2, 0.5)
        
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self._L1 = (
            abs(self._w_h).sum() +
            abs(self._w_h2).sum() +
            abs(self._w_o).sum()
        )
        self._L1_reg= 0.0
        self._L2_reg= 0.001
        # L2 norm ; one regularization option is to enforce
        # L2 norm to be small
        self._L2 = (
            (self._w_h ** 2).sum() +
            (self._w_h2 ** 2).sum() +
            (self._w_o ** 2).sum()
        )
        
        # cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
        # delta = ((Reward.reshape((-1, 1)) + (self._discount_factor * T.max(self.model(ResultState), axis=1, keepdims=True)) ) - self.model(State))
        delta = ((Reward + (self._discount_factor * 
                            T.max(self.model(ResultState, self._w_h_old, self._b_h_old, self._w_h2_old, self._b_h2_old, self._w_o_old, self._b_o_old, 0.2, 0.5), axis=1, keepdims=True)) ) - 
                            (self.model(State, self._w_h, self._b_h, self._w_h2, self._b_h2, self._w_o, self._b_o, 0.2, 0.5))[Action])
        # bellman_cost = T.mean( 0.5 * ((delta) ** 2 ))
        bellman_cost = T.mean( 0.5 * ((delta) ** 2 )) + ( self._L2_reg * self._L2) + ( self._L1_reg * self._L1)

        params = [self._w_h, self._b_h, self._w_h2, self._b_h2, self._w_o, self._b_o]
        updates = sgd(bellman_cost, params, lr=self._learning_rate)
        # updates = RMSprop(bellman_cost, params, lr=self._learning_rate)
        
        self._train = theano.function(inputs=[State, Action, Reward, ResultState], outputs=bellman_cost, updates=updates, allow_input_downcast=True)
        self._predict = theano.function(inputs=[State], outputs=y_pred, allow_input_downcast=True)
        self._q_values = theano.function(inputs=[State], outputs=py_x, allow_input_downcast=True)
        self._bellman_error = theano.function(inputs=[State, Action, Reward, ResultState], outputs=delta, allow_input_downcast=True)
        
        
    def model(self, State, w_h, b_h, w_h2, b_h2, w_o, b_o, p_drop_input, p_drop_hidden):
        State = dropout(State, p_drop_input)
        h = rectify(T.dot(State, w_h) + b_h)
    
        h = dropout(h, p_drop_hidden)
        h2 = rectify(T.dot(h, w_h2) + b_h2)
    
        h2 = dropout(h2, p_drop_hidden)
        q_val_x = T.tanh(T.dot(h2, w_o) + b_o)
        # q_val_x = T.nnet.sigmoid(T.dot(h2, w_o) + b_o)
        return q_val_x
    
    def updateTargetModel(self):
        print "Updating target Model"
        self._w_h_old = self._w_h 
        self._b_h_old = self._b_h 
        self._w_h2_old = self._w_h2
        self._b_h2_old = self._b_h2
        self._w_o_old = self._w_o 
        self._b_o_old = self._b_o 
    
    def train(self, state, action, reward, result_state):
        if (( self._updates % self._weight_update_steps) == 0):
            self.updateTargetModel()
        self._updates += 1
        return self._train(state, action, reward, result_state)
    
    def predict(self, state):
        return self._predict(state)
    def q_values(self, state):
        return self._q_values(state)
    def bellman_error(self, state, action, reward, result_state):
        return self._bellman_error(state, action, reward, result_state)
