import theano
from theano import tensor as T
import numpy as np

# For debugging
# theano.config.mode='FAST_COMPILE'


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.5))

def init_b_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.1), broadcastable=(True, False))

def rectify(X):
    # return X
    return T.maximum(X, 0.)

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

def rlTD(cost, delta, params, lr=0.05):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p + (lr * delta * g)])
    return updates

def RMSpropRL(cost, delta, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.0)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - (lr * delta * g)))
    return updates


class NeuralNet(object):
    
    def __init__(self, input, n_in, n_out):

        hidden_size=36
        batch_size=32
        # self._w_h = init_tanh(n_in, hidden_size)
        self._w_h = init_weights((n_in, hidden_size))
        self._b_h = init_b_weights((1,hidden_size))
        self._w_o = init_weights((hidden_size, n_out))
        self._b_o = init_b_weights((1,n_out))
        
        # self.updateTargetModel()
        
        self._w_h_old = init_tanh(n_in, hidden_size)
        self._w_h_old = init_weights((n_in, hidden_size))
        self._b_h_old = init_b_weights((1,hidden_size))
        self._w_o_old = init_weights((hidden_size, n_out))
        self._b_o_old = init_b_weights((1,n_out))
        
        print "Initial W_h " + str(self._w_h.get_value())
        print "Initial W_o " + str(self._w_o.get_value()) 
        
        self._learning_rate = 0.001
        self._discount_factor= 0.8
        
        self._weight_update_steps=10000
        self._updates=0
        
        
        # data types for model
        State = T.dmatrix("State")
        State.tag.test_value = np.random.rand(batch_size,2)
        ResultState = T.dmatrix("ResultState")
        ResultState.tag.test_value = np.random.rand(batch_size,2)
        Reward = T.col("Reward")
        Reward.tag.test_value = np.random.rand(batch_size,1)
        Action = T.icol("Action")
        Action.tag.test_value = np.zeros((batch_size,1),dtype=np.dtype('int32'))
        # Q_val = T.fmatrix()
        
        self._L1 = (
            abs(self._w_h).sum() +
            abs(self._w_o).sum()
        )
        self._L1_reg= 0.0
        self._L2_reg= 0.001
        # L2 norm ; one regularization option is to enforce
        # L2 norm to be small
        self._L2 = (
            (self._w_h ** 2).sum() +
            (self._w_o ** 2).sum()
        )
        
        # model = T.nnet.sigmoid(T.dot(State, self._w) + self._b.reshape((1, -1)))
        # self._model = theano.function(inputs=[State], outputs=model, allow_input_downcast=True)
        py_x = self.model(State, self._w_h, self._b_h, self._w_o, self._b_o)
        y_pred = T.argmax(py_x, axis=1)
        q_val = T.max(py_x)
        q_func = T.mean((self.model(State, self._w_h, self._b_h, self._w_o, self._b_o))[T.arange(batch_size), Action.reshape((-1,))].reshape((-1, 1)))
        
        # cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
        # delta = ((Reward.reshape((-1, 1)) + (self._discount_factor * T.max(self.model(ResultState), axis=1, keepdims=True)) ) - self.model(State))
        delta = ((Reward + (self._discount_factor * 
                            T.max(self.model(ResultState, self._w_h_old, self._b_h_old, self._w_o_old, self._b_o_old), axis=1, keepdims=True)) ) - 
                            (self.model(State, self._w_h, self._b_h, self._w_o, self._b_o))[T.arange(batch_size), Action.reshape((-1,))].reshape((-1, 1)))
        # bellman_cost = T.mean( 0.5 * ((delta) ** 2 ))
        bellman_cost = T.mean( 0.5 * ((delta) ** 2 )) + ( self._L2_reg * self._L2) + ( self._L1_reg * self._L1)

        params = [self._w_h, self._b_h, self._w_o, self._b_o]
        # updates = sgd(bellman_cost, params, lr=self._learning_rate)
        updates = rlTD(q_func, T.mean(delta), params, lr=self._learning_rate)
        # updates = RMSpropRL(q_func, T.mean(delta), params, lr=self._learning_rate)
        
        self._train = theano.function(inputs=[State, Action, Reward, ResultState], outputs=bellman_cost, updates=updates, allow_input_downcast=True)
        self._predict = theano.function(inputs=[State], outputs=y_pred, allow_input_downcast=True)
        self._q_values = theano.function(inputs=[State], outputs=py_x, allow_input_downcast=True)
        self._bellman_error = theano.function(inputs=[State, Action, Reward, ResultState], outputs=delta, allow_input_downcast=True)
        
        
    def model(self, State, w_h, b_h, w_o, b_o):
        # h = T.tanh(T.dot(State, w_h) + b_h)
        # h = T.dot(State, w_h) + b_h
        
        # (32x2 * 2x36) + 1x36 
        h = rectify(T.dot(State, w_h) + b_h)
        # (32x36 * 36x8) + 1x8)
        qyx = T.tanh(T.dot(h, w_o) + b_o)
        return qyx
    
    def updateTargetModel(self):
        print "Updating target Model"
        self._w_h_old = self._w_h 
        self._b_h_old = self._b_h 
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
