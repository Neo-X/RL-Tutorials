"""
Animation of Elastic collisions with Gravity

author: Jake Vanderplas
email: vanderplas@astro.washington.edu
website: http://jakevdp.github.com
license: BSD
https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
Please feel free to use and modify this, but keep the above information. Thanks!
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
import math
import copy
import collections

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as manimation
from BallGame1D import *
# import scipy.integrate as integrate
# import matplotlib.animation as animation


class BallGame1DState(BallGame1D):

    def __init__(self):
        #------------------------------------------------------------
        # set up initial state
        super(BallGame1DState,self).__init__()
        
        
        
    def getState(self):
        grandularity = 20 # This should be an even number
        action_dimension=1
        range = 1.0
        state = np.zeros(grandularity+1)
        # state[0] = self._box.state[0,1]
        state[0] = self._box.state[0,3]
        
        # state = np.zeros(grandularity)
        delta = self._target[1] - self._previous_max_y
        # print "First delta: " + str(delta)
        delta = (delta)/(range/(grandularity/2.0))
        # print "Second delta: " + str(delta)
        index = int(delta+(grandularity/2)) + action_dimension
        # print "Index is: " + str(index)
        if (index < 1):
            index = 1
        if (index >= len(state)):
            index = len(state)-1
        state[index] = 1.0
        
        return state
    
    def setState(self, st):
        self._agent = st
        self._box.state[0,0] = st[0]
        self._box.state[0,1] = st[1]
        
        
#ani = animation.FuncAnimation(fig, animate, frames=600,
#                               interval=10, blit=True, init_func=init)


# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#ani.save('particle_box.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

# plt.show()

if __name__ == '__main__':
    
    np.random.seed(seed=10)
    game = BallGame1DState()

    game.enableRender()
    game._simulate=True
    # game._saveVideo=True
    print "dt: " + str(game._dt)
    print "BOX: " + str(game._box)
    game.init(np.random.rand(16,16),np.random.rand(16,16),np.random.rand(16,16))
    
    game.reset()
    game.setTarget(np.array([2,2]))
    num_actions=10
    scaling = 2.0
    game._box.state[0][1] = 0
    game.resetTarget()
    game.resetHeight()
    
    actions = (np.random.rand(num_actions,1)-0.5) * 2.0 * scaling
    for action in actions:
        # game.resetTarget()
        state = game.getState()
        print "State: " + str(state)
        print "Action: " + str(action)
        reward = game.actContinuous(action)
        print "Reward: " + str(reward)
        game.resetTarget()
        game.resetHeight()

    game.finish()