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
from BallGame1DChoice import BallGame1DChoice
# import scipy.integrate as integrate
# import matplotlib.animation as animation


class BallGame2DChoice(BallGame1DChoice):

    def __init__(self):
        #------------------------------------------------------------
        # set up initial state
        self._x_v=1.0
        self._x_diff=0.0
        super(BallGame2DChoice,self).__init__()
        

            
    def animate(self, i):
        """perform animation step"""
        out = super(BallGame1DChoice,self).animate(i)
        """step once by dt seconds"""
        
        # update positions
        
        targets_ = np.array(self._targets)
        # print "step targets: " + str(targets_)
        
        targets_[:,:,0] += self._dt * self._x_v * -1.0
        self._targets = collections.deque(list(targets_))
        
        scale=1.0
        ms = int(self._fig.dpi * scale * self._box.size * self._fig.get_figwidth()
                 / np.diff(self._map_ax.get_xbound())[0])
        
        self._plot_target.set_data(targets_[:,:,0].reshape((1,-1))[0], targets_[:,:,1].reshape((1,-1))[0])
        self._plot_target.set_markersize(ms)
        
        self._plot_target_choice.set_data([targets_[0,self._target_choice,0]], [targets_[0,self._target_choice,1]])
        self._plot_target_choice.set_markersize(ms)

        # return particles, rect
        return out

    def actContinuous(self, action):
        run = True
        
        v = self._box.state[0][3] + action[0]
        time = self._computeTime(v)
        targets_ = np.array(self._targets)
        self._x_diff = (time*self._x_v) - targets_[0,0,0] + 2.0
        # print "Diff: " +str(diff)
        # x direction is 1/s
        self._x_v = action[1]
        # print "Acting: " + str(action)
        # self._box.state[0][2] = action[0]
        self._box.state[0][3] += action[0]
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
            # print "self._max_y: " + str(self._max_y)
        return self.reward()
    
    def reward(self):
        # More like a cost function for distance away from target
        d = math.fabs(self._max_y - self._target[1]) + math.fabs(self._x_diff)
        return -d
        
                
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
    ballGame = BallGame2DChoice()

    ballGame.enableRender()
    ballGame._simulate=True
    # ballGame._saveVideo=True
    print "dt: " + str(ballGame._dt)
    print "BOX: " + str(ballGame._box)
    ballGame.init(np.random.rand(16,16),np.random.rand(16,16),np.random.rand(16,16))
    
    ballGame.reset()
    ballGame.setTarget(np.array([2,2]))
    num_actions=10
    scaling = 2.0
    ballGame._box.state[0][1] = 0
    
    actions = (np.random.rand(num_actions,1)-0.5) * 2.0 * scaling
    for action in actions:
        # ballGame.resetTarget()
        state = ballGame.getState()
        print "State: " + str(state)
        print "Action: " + str(action)
        ballGame.setTargetChoice(0)
        reward = ballGame.actContinuous(action)
        print "Reward: " + str(reward)
        # print "targets: " + str(ballGame._targets)
        ballGame.resetTarget()

    ballGame.finish()