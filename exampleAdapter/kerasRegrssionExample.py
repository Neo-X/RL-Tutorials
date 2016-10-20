'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
# from model.ModelUtil import *
import itertools
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense, Dropout, Activation
import random
import example

np.random.seed(1337)  # for reproducibility

def f(x):
    return (math.cos(x)-0.75)*(math.sin(x)+0.75)

if __name__ == '__main__':
    batch_size = 32
    nb_epoch = 10
    
    function = example.Function()

    state_bounds = np.array([[0.0],[5.0]])
    action_bounds = np.array([[-4.0],[2.0]])
    function.setStateBounds(state_bounds)
    function.setActionBounds(action_bounds)
    experience_length = 200
    batch_size=32
    # states = np.repeat(np.linspace(0.0, 5.0, experience_length),2, axis=0)
    states = np.linspace(0.0, 5.0, experience_length)
    states = np.reshape(states, (len(states), 1))
    shuffle = range(experience_length)
    states = states[shuffle]
    # random.shuffle(shuffle)
    print ("States: " , states)
    old_states = states
    norm_states=[]
    norm_states = list(map(function.norm_state, states))
    X_test = X_train = norm_states
    
    # states = np.linspace(-5.0,-2.0, experience_length/2)
    # states = np.append(states, np.linspace(-1.0, 5.0, experience_length/2))
    # print states
    # actions = np.array(map(fNoise, states))
    actions = np.reshape(np.array(list(map(function.func, states))), (len(states), 1))
    # actions = actions[shuffle]
    print ("Actions: ", actions)
    norm_actions = np.array(list(map(function.norm_action, actions)))
    y_train = y_test = norm_actions
    settings = {}
    print("X_test:", X_test)
    print("Y_test:", y_test)
    
    model = Sequential()
    # 2 inputs, 10 neurons in 1 hidden layer, with tanh activation and dropout
    model.add(Dense(128, init='uniform', input_shape=(1,))) 
    model.add(Activation('relu'))
    model.add(Dense(64, init='uniform')) 
    model.add(Activation('relu'))
    # 1 output, linear activation
    model.add(Dense(1, init='uniform'))
    model.add(Activation('linear'))
    
    sgd = SGD(lr=0.01, momentum=0.9)
    print ("Clipping: ", sgd.decay)
    model.compile(loss='mse', optimizer=sgd)
    
    from keras.callbacks import EarlyStopping
    # early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    
    errors=[]
    
    score = model.fit(X_train, y_train,
              nb_epoch=156, batch_size=32,
              validation_data=(X_test, y_test)
              # callbacks=[early_stopping],
              )
    
    errors.extend(score.history['loss'])
    
    
    print ("Score: " , errors)
    
    
    states = (np.linspace(-1.0, 6.0, experience_length))
    states = np.reshape(states, (len(states), 1))
    norm_states = np.array(list(map(function.norm_state, states)))
    # norm_states = ((states  - 2.5)/3.5)
    # predicted_actions = np.array(map(model.predict, states))
    
    # x=np.transpose(np.array([states]))
    
    # print ("States: " , x)
    norm_predicted_actions = np.array(model.predict(norm_states, batch_size=200, verbose=0), dtype='float64')
    norm_predicted_actions = np.reshape(norm_predicted_actions, (len(states), 1))
    print ("norm_predicted_actions[0]: ", norm_predicted_actions[0])
    p_a = function.scale_action(norm_predicted_actions[0])
    print ("norm_predicted_actions: ", norm_predicted_actions)
    predicted_actions = np.array(list(map(function.scale_action, norm_predicted_actions)))
    predicted_actions = np.reshape(predicted_actions, (len(states), 1))
    # print ("Prediction: ", predicted_actions)
    
    # print "var : " + str(predicted_actions_var)
    # print "act : " + str(predicted_actions)
    
    
    std = 1.0
    # _fig, (_bellman_error_ax, _reward_ax, _discount_error_ax) = plt.subplots(1, 1, sharey=False, sharex=True)
    _fig, (_bellman_error_ax, _training_error_ax) = plt.subplots(1, 2, sharey=False, sharex=False)
    _bellman_error, = _bellman_error_ax.plot(old_states, actions, linewidth=2.0, color='y', label="True function")
    # _bellman_error, = _bellman_error_ax.plot(states, predicted_actions_dropout, linewidth=2.0, color='r', label="Estimated function with dropout")
    _bellman_error, = _bellman_error_ax.plot(states, predicted_actions, linewidth=2.0, color='g', label="Estimated function")
    # _bellman_error, = _bellman_error_ax.plot(states, actionsNoNoise, linewidth=2.0, label="True function")
    
    
    # _bellman_error_std = _bellman_error_ax.fill_between(states, predicted_actions - predicted_actions_var,
    #                                                     predicted_actions + predicted_actions_var, facecolor='green', alpha=0.5)
    # _bellman_error_std = _bellman_error_ax.fill_between(states, lower_var, upper_var, facecolor='green', alpha=0.5)
    # _bellman_error_ax.set_title("True function")
    _bellman_error_ax.set_ylabel("function value: f(x)")
    _bellman_error_ax.set_xlabel("x")
    # Now add the legend with some customizations.
    legend = _bellman_error_ax.legend(loc='lower right', shadow=True)
    _bellman_error_ax.set_title("Predicted curves")
    _bellman_error_ax.grid(b=True, which='major', color='black', linestyle='-')
    _bellman_error_ax.grid(b=True, which='minor', color='g', linestyle='--')
    
    
    """
    _reward, = _reward_ax.plot([], [], linewidth=2.0)
    _reward_std = _reward_ax.fill_between([0], [0], [1], facecolor='blue', alpha=0.5)
    _reward_ax.set_title('Mean Reward')
    _reward_ax.set_ylabel("Reward")
    _discount_error, = _discount_error_ax.plot([], [], linewidth=2.0)
    _discount_error_std = _discount_error_ax.fill_between([0], [0], [1], facecolor='blue', alpha=0.5)
    _discount_error_ax.set_title('Discount Error')
    _discount_error_ax.set_ylabel("Absolute Error")
    plt.xlabel("Iteration")
    """
    _title = "Training function"
    _fig.suptitle(_title, fontsize=18)
    
    _fig.set_size_inches(8.0, 4.5, forward=True)
    # er = plt.figure(2)
    _training_error_ax.plot(range(len(errors)), errors)
    _training_error_ax.set_ylabel("Error")
    _training_error_ax.set_xlabel("Iteration")
    _training_error_ax.set_title("Training Error")
    _training_error_ax.grid(b=True, which='major', color='black', linestyle='-')
    _training_error_ax.grid(b=True, which='minor', color='g', linestyle='--')
    
    # _fig.show()
    # er.show()
    plt.show()
    
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])