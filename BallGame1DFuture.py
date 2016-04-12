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


class BallGame1DFuture(BallGame1D):

    def __init__(self):
        #------------------------------------------------------------
        # set up initial state
        super(BallGame1DFuture,self).__init__()
        
        

    def reset(self):
        self._box.state[0][0] = 2.0
        self._box.state[0][1] = self._box.bounds[2]+0.1
        self._box.state[0][2] = 0
        self._box.state[0][3] = (np.random.rand(1)+6.2) # think this will be about middle, y = 2.0
        num_future_targets=3
        self._targets = collections.deque(list( ((np.random.rand(num_future_targets)-0.5) * 2.0) + 2))
        self.setTarget(np.array([2,self._targets[0]]))
        
    def resetTarget(self):
        """
        y range is [1,3]
        """
        self._targets.append( ((np.random.rand(1)-0.5) * 2.0) + 2)
        self._targets.popleft()
        self.setTarget(np.array([2,self._targets[0]]))
        
    def getState(self):
        state = np.array([0.0,0.0,0.0,0.0], dtype=float)
        # state[0] = self._box.state[0,1]
        state[0] = self._targets[0]
        state[1] = self._box.state[0,3]
        state[2] = self._targets[1]
        state[3] = self._targets[2]
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
    ballGame = BallGame1DFuture()

    ballGame.enableRender()
    ballGame._simulate=True
    # ballGame._saveVideo=True
    print "dt: " + str(ballGame._dt)
    print "BOX: " + str(ballGame._box)
    ballGame.init(np.random.rand(256,1),np.random.rand(256,1),np.random.rand(256,1))
    
    ballGame.reset()
    ballGame.setTarget(np.array([2,2]))
    num_actions=10
    scaling = 2.0
    ballGame._box.state[0][1] = 0
    
    actions = (np.random.rand(num_actions,1)-0.5) * 2.0 * scaling
    for action in actions:
        # ballGame.resetTarget()
        state = ballGame.getState()
        # print "State: " + str(state)
        print "Action: " + str(action)
        reward = ballGame.actContinuous(action)
        print "Reward: " + str(reward)

    ballGame.finish()