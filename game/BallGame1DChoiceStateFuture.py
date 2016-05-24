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

from BallGame1DChoice import BallGame1DChoice
# import scipy.integrate as integrate
# import matplotlib.animation as animation


class BallGame1DChoiceStateFuture(BallGame1DChoice):

    def __init__(self):
        #------------------------------------------------------------
        # set up initial state
        super(BallGame1DChoiceStateFuture,self).__init__()
        self._granularity = 20 # This should be an even number
        self._action_dimension=1
        self._range = 1.0
        self._future = 3

    
    def getState(self):
        state = np.zeros((self._granularity*self._future)+self._action_dimension)
        # state[0] = self._box.state[0,1]
        state[0] = self._box.state[0,3]
        # print "sSelf: " + str(self._choices)
        for f in range(self._future):
            
            for i in range(int(self._choices)):
                # state = np.zeros(grandularity)
                delta = self._targets[0][i][1] - self._previous_max_y
                # print "First delta: " + str(delta)
                delta = (delta)/(self._range/(self._granularity/2.0))
                # print "Second delta: " + str(delta)
                index = int(delta+(self._granularity/2))
                # print "Index is: " + str(index)
                if (index < 1):
                    index = 0
                if (index >= self._granularity):
                    index = self._granularity-1
                state[(f*self._granularity)+index+self._action_dimension] = 1.0
        
        return state
    
    def setState(self, st):
        self._agent = st
        self._box.state[0,0] = st[0]
        self._box.state[0,1] = st[1]
        
    def setTargetChoice(self, i):
        # need to find which target corresponds to this bin.
        _loc = np.linspace(-self._range, self._range, self._granularity)[i]
        min_dist = 100000000.0
        _choice = -1
        for i in range(int(self._choices)):
            _target_loc = self._targets[0][i][1]
            _tmp_dist = math.fabs(_target_loc - _loc)
            if ( _tmp_dist < min_dist):
                _choice = i
                min_dist = _tmp_dist
        self._target_choice = i
        self._target = self._targets[0][i]
        
        
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
    game = BallGame1DChoiceStateFuture()

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