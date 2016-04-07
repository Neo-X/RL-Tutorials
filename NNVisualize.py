import matplotlib.pyplot as plt
# from matplotlib import mpl
import numpy as np
# import matplotlib.animation as animation
import random
import sys
import json

class NNVisualize(object):
    
    def __init__(self, title):
        """
            Three plots
            bellman error
            average reward
            discounted reward error
        """
        
        self._title=title
        
        """
        self._fig, (self._bellman_error_ax, self._reward_ax, self._discount_error_ax) = plt.subplots(3, 1, sharey=False, sharex=True)
        self._bellman_error, = self._bellman_error_ax.plot([], [], linewidth=2.0)
        self._bellman_error_std = self._bellman_error_ax.fill_between([0], [0], [1], facecolor='blue', alpha=0.5)
        self._bellman_error_ax.set_title('Bellman Error')
        self._bellman_error_ax.set_ylabel("Absolute Error")
        self._reward, = self._reward_ax.plot([], [], linewidth=2.0)
        self._reward_std = self._reward_ax.fill_between([0], [0], [1], facecolor='blue', alpha=0.5)
        self._reward_ax.set_title('Mean Reward')
        self._reward_ax.set_ylabel("Reward")
        self._discount_error, = self._discount_error_ax.plot([], [], linewidth=2.0)
        self._discount_error_std = self._discount_error_ax.fill_between([0], [0], [1], facecolor='blue', alpha=0.5)
        self._discount_error_ax.set_title('Discount Error')
        self._discount_error_ax.set_ylabel("Absolute Error")
        plt.xlabel("Iteration")
        
        self._fig.set_size_inches(8.0, 12.5, forward=True)
        """
        
    def init(self):
        """
            Three plots
            bellman error
            average reward
            discounted reward error
        """
        # self._fig, (self._bellman_error_ax, self._reward_ax, self._discount_error_ax) = plt.subplots(1, 1, sharey=False, sharex=True)
        self._fig, (self._bellman_error_ax) = plt.subplots(1, 1, sharey=False, sharex=True)
        self._bellman_error, = self._bellman_error_ax.plot([], [], linewidth=2.0)
        self._bellman_error_std = self._bellman_error_ax.fill_between([0], [0], [1], facecolor='blue', alpha=0.5)
        self._bellman_error_ax.set_title('Error')
        self._bellman_error_ax.set_ylabel("Absolute Error")
        """
        self._reward, = self._reward_ax.plot([], [], linewidth=2.0)
        self._reward_std = self._reward_ax.fill_between([0], [0], [1], facecolor='blue', alpha=0.5)
        self._reward_ax.set_title('Mean Reward')
        self._reward_ax.set_ylabel("Reward")
        self._discount_error, = self._discount_error_ax.plot([], [], linewidth=2.0)
        self._discount_error_std = self._discount_error_ax.fill_between([0], [0], [1], facecolor='blue', alpha=0.5)
        self._discount_error_ax.set_title('Discount Error')
        self._discount_error_ax.set_ylabel("Absolute Error")
        plt.xlabel("Iteration")
        """
        self._fig.suptitle(self._title, fontsize=18)
        
        self._fig.set_size_inches(8.0, 4.5, forward=True)
        
    def updateLoss(self, error, std):
        self._bellman_error.set_xdata(np.arange(len(error)))
        self._bellman_error.set_ydata(error)
        self._bellman_error_ax.collections.remove(self._bellman_error_std)
        self._bellman_error_std = self._bellman_error_ax.fill_between(np.arange(len(error)), error - std, error + std, facecolor='blue', alpha=0.5)
        
        
        self._bellman_error_ax.relim()      # make sure all the data fits
        self._bellman_error_ax.autoscale()
        
        
    def show(self):
        plt.show()
        
    def redraw(self):
        self._fig.canvas.draw()
        
    def setInteractive(self):
        plt.ion()
        
    def setInteractiveOff(self):
        plt.ioff()
        
    def saveVisual(self, fileName):
        plt.savefig(fileName+".svg")
        
if __name__ == "__main__":
    
    datafile = sys.argv[1]
    file = open(datafile)
    trainData = json.load(file)
    # print "Training data: " + str(trainingData)
    file.close()
    
    """
    trainData["mean_reward"]=[]
    trainData["std_reward"]=[]
    trainData["mean_bellman_error"]=[]
    trainData["std_bellman_error"]=[]
    trainData["mean_discount_error"]=[]
    trainData["std_discount_error"]=[]
    
    """
    
    rlv = NNVisualize()
    rlv.updateLoss(np.array(trainData["mean_bellman_error"]), np.array(trainData["std_bellman_error"]))
    rlv.show()