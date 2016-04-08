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

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as manimation
# import scipy.integrate as integrate
# import matplotlib.animation as animation

class ParticleLine:
    """Orbits class
    
    init_state is an [N x 4] array, where N is the number of particles:
       [[x1, y1, vx1, vy1],
        [x2, y2, vx2, vy2],
        ...               ]

    bounds is the size of the box: [xmin, xmax, ymin, ymax]
    """
    def __init__(self,
                 init_state = [[1, 0, 0, -1],
                               [-0.5, 0.5, 0.5, 0.5],
                               [-0.5, -0.5, -0.5, 0.5]],
                 bounds = [0, 4, 0, 4],
                 size = 0.04,
                 M = 1.0,
                 G = 9.8):
        self.init_state = np.array(init_state, dtype=float)
        self.M = M * np.ones(self.init_state.shape[0])
        self.size = size
        self.state = self.init_state.copy()
        self.time_elapsed = 0
        self.bounds = bounds
        self.G = G

    def step(self, dt):
        """step once by dt seconds"""
        self.time_elapsed += dt
        
        # update positions
        self.state[:, :2] += dt * self.state[:, 2:]


        """
            TODO: Might be an issue where particle can not go outside boundary
        """
        # check for crossing boundary
        crossed_x1 = (self.state[:, 0] < self.bounds[0] + self.size)
        crossed_x2 = (self.state[:, 0] > self.bounds[1] - self.size)
        crossed_y1 = (self.state[:, 1] < self.bounds[2] + self.size)
        crossed_y2 = (self.state[:, 1] > self.bounds[3] - self.size)

        self.state[crossed_x1, 0] = self.bounds[0] + self.size
        self.state[crossed_x2, 0] = self.bounds[1] - self.size

        self.state[crossed_y1, 1] = self.bounds[2] + self.size
        self.state[crossed_y2, 1] = self.bounds[3] - self.size

        # self.state[crossed_x1 | crossed_x2, 2] *= -1
        # self.state[crossed_y1 | crossed_y2, 3] *= -1
        self.state[crossed_y1, 3] *= -1
        if (crossed_y1[0] ):
            # Crosses ground boundary
            # print "Bounce Ground"
            # print crossed_y1, crossed_y2
            return False

        # add gravity
        self.state[:, 3] -= self.M * self.G * dt
        return True


class BallGame1D(object):

    def __init__(self):
        #------------------------------------------------------------
        # set up initial state
        np.random.seed(0)
        init_state = -0.5 + np.random.random((50, 4))
        init_state[:, :2] *= 3.9
        # [y, v_y]
        init_state = [[2, 0.1,0, 1]]
        
        self._box = ParticleLine(init_state, size=0.04)
        self._dt = 1. / 30 # 30fps
        self._max_y = init_state[0][1]
        self._render=False
        self._simulate=True
        self._saveVideo=False
        
        

    def reset(self):
        self._box.state[0][0] = 2.0
        self._box.state[0][1] = self._box.bounds[2]+0.1
        self._box.state[0][2] = 0
        self._box.state[0][3] = (np.random.rand(1)+6.2) # think this will be about middle, y = 2.0
        self.setTarget(np.array([2,((np.random.rand(1)-0.5) * 2.0) + 2]))
        
    def resetTarget(self):
        """
        y range is [1,3]
        """
        self.setTarget(np.array([2,((np.random.rand(1)-0.5) * 2.0) + 2]))
        
    def move(self, action):
        """
        action in [0,1,2,3,4,5,6,7]
        Used for initial bootstrapping
        """
        return {
            0: [-1,1],
            1: [-0.8,1],
            2: [-0.66,1],
            3: [-0.33,1],
            4: [0.0,1],
            5: [0.33,1],
            6: [0.66,1],
            7: [1,1],
            }.get(action, [-1,0]) 
            
    def setMovieName(self, name):
        self._movie_name=name
        
    def init(self, U, V, Q):
        """initialize animation"""
         #------------------------------------------------------------
        # set up figure and animation
        self._fig, (self._map_ax, self._policy_ax) = plt.subplots(1, 2, sharey=False)
        # self._fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self._fig.set_size_inches(12.5, 6.5, forward=True)
        self._map_ax.set_title('Map')
        
        # particles holds the locations of the particles
        self._particles, = self._map_ax.plot([0,4], [0,4], 'bo', ms=4)
        self._targets, = self._map_ax.plot([], [], 'go', ms=4)
        
        # rect is the box edge
        self._rect = plt.Rectangle(self._box.bounds[::2],
                             self._box.bounds[1] - self._box.bounds[0],
                             self._box.bounds[3] - self._box.bounds[2],
                             ec='none', lw=2, fc='none')
        self._map_ax.add_patch(self._rect)
        
        self._policy_ax.set_title('Policy')
        
        scale =float(4.0)
        X,Y = np.mgrid[0:self._box.bounds[1]*scale,0:self._box.bounds[1]*scale]/float(scale)
        # print X,Y
        # self._policy = self._policy_ax.quiver(X[::2, ::2],Y[::2, ::2],U[::2, ::2],V[::2, ::2], linewidth=0.5, pivot='mid', edgecolor='k', headaxislength=5, facecolor='None')
        textstr = """$\max q=%.2f$\n$\min q=%.2f$"""%(np.max(Q), np.min(Q))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
        
        # place a text box in upper left in axes coords
        self._policyText = self._policy_ax.text(0.05, 0.95, textstr, transform=self._policy_ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        q_max = np.max(Q)
        q_min = np.min(Q)
        Q = (Q - q_min)/ (q_max-q_min)
        self._policy2 = self._policy_ax.quiver(X,Y,U,V,Q, alpha=.75, linewidth=1.0, pivot='mid', angles='xy', linestyles='-', scale=25.0)
        self._policy = self._policy_ax.quiver(X,Y,U,V, linewidth=0.5, pivot='mid', edgecolor='k', headaxislength=3, facecolor='None', angles='xy', linestyles='-', scale=25.0)
        
        # self._policy_ax.set_aspect(1.)
        self.setTarget(np.array([2,0]))
        
        self._particles.set_data([], [])
        self._rect.set_edgecolor('none')
        plt.ion()
        plt.show()
        
        if self._saveVideo:
            FFMpegWriter = manimation.writers['ffmpeg']
            metadata = dict(title='Movie Test', artist='Matplotlib',
                            comment='Movie support!')
            self._writer = FFMpegWriter(fps=30, metadata=metadata)
            self._writer.setup(self._fig, str(self._movie_name) + ".mp4", 100)
            # self._movieOut = open("moviee.mp4", 'wb')
            # self._writer.fig = self._fig
            # self._writer.fig = self._fig
        return self._particles, self._rect
    
    def animate(self, i):
        """perform animation step"""
        out = self._box.step(self._dt)
    
        ms = int(self._fig.dpi * 2 * self._box.size * self._fig.get_figwidth()
                 / np.diff(self._map_ax.get_xbound())[0])
        
        # update pieces of the animation
        self._rect.set_edgecolor('k')
        self._particles.set_data(self._box.state[:, 0], self._box.state[:, 1])
        self._particles.set_markersize(ms)
        self._targets.set_data([self._target[0]], [self._target[1]])
        self._targets.set_markersize(ms)
        # return particles, rect
        return out
    
        
    def actContinuous(self, action):
        run = True
        # print "Acting: " + str(action)
        # self._box.state[0][2] = action[0]
        self._box.state[0][3] += action[0]
        self._box.state[0][1] =0
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
                    return self.reward()
        else:
            # self._max_y = self._box.state[0][1]
            self._max_y = self._computeHeight(action_=self._box.state[0][3])
            # print "self._max_y: " + str(self._max_y)
        return self.reward()
    
    def _computeHeight(self, action_):
        init_v_squared = (action_*action_)
        seconds_ = 2 * (-self._box.G)
        return (-init_v_squared)/seconds_
            
    def reward(self):
        # More like a cost function for distance away from target
        d = math.fabs(self._max_y - self._target[1])
        self._max_y = self._box.state[0][1]
        return -d
    
    def update(self):
        """perform animation step"""
        # update pieces of the animation
        # self._agent = self._agent + np.array([0.1,0.1])
        # print "Agent loc: " + str(self._agent)
        self._fig.canvas.draw()
        if self._saveVideo:
            self._writer.grab_frame()
            # self._fig.savefig(self._movieOut, format='rgba',
                # dpi=100)
        # self._line1.set_ydata(np.sin(x + phase))
        # self._fig.canvas.draw()
        
    def updatePolicy(self, U, V, Q):
                # self._policy.set_UVC(U[::2, ::2],V[::2, ::2])
        textstr = """$\max q=%.2f$\n$\min q=%.2f$"""%(np.max(Q), np.min(Q))
        self._policyText.set_text(textstr)
        q_max = np.max(Q)
        q_min = np.min(Q)
        Q = (Q - q_min)/ (q_max-q_min)
        self._policy2.set_UVC(U, V, Q)
        # self._policy2.set_vmin(1.0)
        """
        self._policy2.update_scalarmappable()
        print "cmap " + str(self._policy2.cmap)  
        print "Face colours" + str(self._policy2.get_facecolor())
        colours = ['gray','black','blue']
        cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                                   colours,
                                                   256)
        self._policy2.cmap._set_extremes()
        """
        self._policy.set_UVC(U, V)
        self._fig.canvas.draw()
    
    def getState(self):
        state = np.array([0.0,0.0], dtype=float)
        # state[0] = self._box.state[0,1]
        state[0] = self._target[1] - state[1]
        state[1] = self._box.state[0][3]
        return state
    
    def setState(self, st):
        self._agent = st
        self._box.state[0][0] = st[0]
        self._box.state[0][1] = st[1]
        
    def setTarget(self, st):
        self._target = st
        
    def reachedTarget(self):
        # Might be a little touchy because floats are used
        return False
    
    def finish(self):
        plt.ioff()
        # self._movieOut.close()
        if self._writer:
            self._writer.finish()
        

    def saveVisual(self, fileName):
        # plt.savefig(fileName+".svg")
        self._fig.savefig(fileName+".svg")
        
    def enableRender(self):
        self._render=True
    def disableRender(self):
        self._render=False

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
    ballGame = BallGame1D()

    ballGame.enableRender()
    ballGame._simulate=True
    # ballGame._saveVideo=True
        
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
        print "State: " + str(state)
        print "Action: " + str(action)
        reward = ballGame.actContinuous(action)
        print "Reward: " + str(reward)

    ballGame.finish()