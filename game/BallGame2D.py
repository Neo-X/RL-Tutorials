"""
    
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
from BallGame1D import BallGame1D 
# import scipy.integrate as integrate
# import matplotlib.animation as animation


class BallGame2D(BallGame1D):

    def __init__(self):
        #------------------------------------------------------------
        # set up initial state
        self._x_v=1.0
        self._x_diff=0.0
        super(BallGame2D,self).__init__()
        
        
        
    def reset(self):
        self._box.state[0][0] = 2.0
        self._box.state[0][1] = self._box.bounds[2]+0.1
        self._box.state[0][2] = 0
        self._box.state[0][3] = (np.random.rand(1)+self._defaultVelocity) # think this will be about middle, y = 2.0
        self.resetTarget()
            
    def animate(self, i):
        """perform animation step"""
        out = super(BallGame2D,self).animate(i)
        """step once by dt seconds"""
        
        # update positions
        
        
        self._target[0] += self._dt * self._x_v * -1.0
        
        scale=1.0
        ms = int(self._fig.dpi * scale * self._box.size * self._fig.get_figwidth()
                 / np.diff(self._map_ax.get_xbound())[0])
        
        self._plot_target.set_data([self._target[0]], [self._target[1]])
        self._plot_target.set_markersize(ms)
        
        # self._plot_target_choice.set_data([target[0]], [target[1]])
        # self._plot_target_choice.set_markersize(ms)

        # return particles, rect
        return out
    
    def actContinuous(self, action):
        run = True
        self._box.state[0][3] += action[0]
        self._x_v = action[1]
        time = self._computeTime(self._box.state[0][3])
        target_ = np.array(self._target)
        self._x_diff = ((time*self._x_v)+2.0) - target_[0]
        # print "Diff: " +str(diff)
        # x direction is 1/s
        # print "Acting: " + str(action)
        # self._box.state[0][2] = action[0]
        oldstate = self._box.state[0][3]
        self._box.state[0][1] =0
        # print "New state: " + str(self._box.state[0][3])
        if self._simulate:
            for i in range(500):
                run = self.animate(i)
                # print box.state
                if self._max_y < self._box.state[0][1]:
                    self._max_y = self._box.state[0][1]
                # print "Max_y: " + str(self._max_y)
                if self._render:
                    self.update()
                
                if not run:
                    # print "self._max_y: " + str(self._max_y)
                    self._box.state[0][3] = oldstate # Need to set state to initial to help eliminate errors
                    return self.reward()
        else:
            # self._max_y = self._box.state[0][1]
            self._max_y = self._computeHeight(action_=self._box.state[0][3])
            target_[0] -= (time*self._x_v)
            self.setTarget(target_)
            # print "self._max_y: " + str(self._max_y)
        return self.reward()
    
    def reward(self):
        # More like a cost function for distance away from target
        d = math.fabs(self._max_y - self._target[1]) + math.fabs(self._x_diff)
        # d = math.fabs(self._max_y - self._target[1])
        return -d
         
                
    def resetTarget(self):
        """
        y range is [1,3]
        """
        val=np.array([3.0 + (np.random.rand(1)-0.4)[0],self.generateNextTarget(self._target[1])])
        self.setTarget(val)
    
        
    def getState(self):
        state = []
        state.append(self._target[1] - self._previous_max_y)
        state.append( self._box.state[0,3]) # velocity in y
        state.append( self._x_v) # velocity in x
        state.append(self._target[0] - 2.0)
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
    game = BallGame2D()

    game.enableRender()
    game._simulate=True
    # game._saveVideo=True
    print "dt: " + str(game._dt)
    print "BOX: " + str(game._box)
    game.init(np.random.rand(16,16),np.random.rand(16,16),np.random.rand(16,16))
    
    game.reset()
    # game.resetTarget()
    # game.setTarget(np.array([2,2]))
    num_actions=10
    action_lenth=2
    scaling = 1.0
    game._box.state[0][1] = 0
    
    actions = (np.random.rand(num_actions,action_lenth)-0.5) * 2.0 * scaling
    for action in actions:
        # game.resetTarget()
        state = game.getState()
        print "State: " + str(state)
        action[1] = 1.0
        print "Action: " + str(action)
        reward = game.actContinuous(action)
        print "Reward: " + str(reward)
        # print "targets: " + str(game._targets)
        game.resetTarget()
        game.resetHeight()

    game.finish()