import theano
from theano import tensor as T
import numpy as np

def floatX(State):
    return np.asarray(State, dtype=theano.config.floatX)

def init_weights(shape):
    # return theano.shared(floatX(np.zeros(shape)))
    return theano.shared(floatX(np.random.randn(*shape) * 1.0))

def init_b_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.1))

def init_tanh(n_in, n_out, r_num):
    rng = np.random.RandomState(r_num)
    return theano.shared(np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            ))


class RLLogisticRegression(object):
    """Reinforcement Learning based Logistic regression model

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Value function approximation is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a action value.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        
        # self._w = init_weights((n_in, n_out))
        # self._w_old = init_weights((n_in, n_out))
        self._w = init_tanh(n_in, n_out, 1234)
        self._w_old = init_tanh(n_in, n_out, 2235)
        print "Initial W " + str(self._w.get_value()) 
        # (n_out,) ,) used so that it can be added as row or column
        self._b = init_b_weights((n_out,))
        self._b_old = init_b_weights((n_out,))
        
        # learning rate for gradient descent updates.
        self._learning_rate = 0.005
        # future discount 
        self._discount_factor= 0.8
        self._weight_update_steps=5000
        self._updates=0
        
        # data types for model
        State = T.fmatrix("State")
        ResultState = T.fmatrix("ResultState")
        Reward = T.col("Reward")
        Action = T.icol("Action")
        # Q_val = T.fmatrix()
        
        model = T.tanh(T.dot(State, self._w) + self._b)
        self._model = theano.function(inputs=[State], outputs=model, allow_input_downcast=True)
        
        q_val = self.model(State, self._w, self._b)
        action_pred = T.argmax(q_val, axis=1)
        
        # bellman error, delta error
        delta = ((Reward + (self._discount_factor * T.max(self.model(ResultState, self._w_old, self._b_old), axis=1, keepdims=True)) ) -
                  (self.model(State, self._w, self._b))[Action])
        # delta = ((Reward + (self._discount_factor * T.max(self.model(ResultState), axis=1, keepdims=True)) ) - T.max(self.model(State), axis=1,  keepdims=True))
        
        self._L2_reg= 0.01
        # L2 norm ; one regularization option is to enforce
        # L2 norm to be small
        self._L2 = (
            (self._w** 2).sum()
        )
        # total bellman cost 
        # Squaring is important so errors do not cancel each other out.
        # mean is used instead of sum as it is more independent of parameter scale
        bellman_cost = T.mean(0.5 *  ((delta) ** 2 ) ) + (self._L2 * self._L2_reg)
        
        # Compute gradients w.r.t. model parameters
        gradient = T.grad(cost=bellman_cost, wrt=self._w)
        gradient_b = T.grad(cost=bellman_cost, wrt=self._b)
        
        """
            Updates to apply to parameters
            Performing gradient descent, want to add steps in the negative direction of 
            gradient.
        """
        update = [[self._w, self._w + (-gradient * self._learning_rate)],
                  [self._b, self._b + (-gradient_b * self._learning_rate)]]
        
        # This function performs one training step and update
        self._train = theano.function(inputs=[State, Action, Reward, ResultState], outputs=bellman_cost, updates=update, allow_input_downcast=True)
        # Used to get to predicted actions to select
        self._predict = theano.function(inputs=[State], outputs=action_pred, allow_input_downcast=True)
        self._q_values = theano.function(inputs=[State], outputs=q_val, allow_input_downcast=True)
        self._bellman_error = theano.function(inputs=[State, Action, Reward, ResultState], outputs=delta, allow_input_downcast=True)
        

    def model(self, State, w, b):
        """
        Beter to have only one model function
        """
        # return self._model(State)
        return T.tanh(T.dot(State, w) + b)
        # return T.nnet.sigmoid(T.dot(State, self._w) + self._b)
        # return T.dot(State, self._w) + self._b
        # return (T.dot(State, self._w) + self._b.reshape((1, -1)))
        
    def updateTargetModel(self):
        print "Updating target Model"
        self._w_old = self._w 
        self._b_old = self._b
        
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