# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 15:29:37 2021

@author: Logan
"""

import gym
from gym import spaces

import numpy as np
import math


#material for plotting
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class ParticlePlot:
    
    def __init__(self, title=None, window_size=[10, 10]):
        #defualt to array of None so we can call .all() in render()
        self.last_states = np.array([None, None, None, None])
        self.xlim = (window_size[0], window_size[1])
        self.ylim = (window_size[0], window_size[1])
        #create the plot
        fig = plt.figure()
        fig.suptitle(title)
        #Next, draw a rectangle around the limits
        rect = patches.Rectangle((self.xlim[0],self.ylim[0]),
                                 self.xlim[1]-self.xlim[0],
                                 self.ylim[1]-self.ylim[0],
                                 linewidth=1,edgecolor='r',facecolor='none')
    #    fig.gca().set(xlim=self.xlim, ylim=self.ylim)
        fig.gca().add_patch(rect)
        plt.show(block=False)
        fig.show()
        self.fig = fig;

        
    def render(self, states):
        #plot a red circle at the particle's location
        nextPos = plt.Circle( ( states[0], states[1] ), 0.5, color='r')
        #get the figure and add the circle
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_patch(nextPos)
        if (self.last_states != None).all():
            dx = states[0] - self.last_states[0]
            dy = states[1] - self.last_states[1]
            plt.arrow(self.last_states[0], self.last_states[1], 
                      dx, dy, width=0.05, shape='right')
        
        
        plt.pause(.05)
        self.last_states = states
        
    def reset(self):
        self.last_states = np.array([None, None, None, None])
        plt.clf()
        rect = patches.Rectangle((self.xlim[0],self.ylim[0]),
                                 self.xlim[1]-self.xlim[0],
                                 self.ylim[1]-self.ylim[0],
                                 linewidth=1,edgecolor='r',facecolor='none')
    #    fig.gca().set(xlim=self.xlim, ylim=self.ylim)
        plt.gca().add_patch(rect)
    



    def close(self):
        plt.show(block=True)



class SimpleParticle(gym.Env):
    """The simple particle model adapted to use opengym"""
    metadata = {'render.modes': ['human']}


    def __init__(self, numParticles, phi, stateBounds, dwellTime, maxSteps):
        super(SimpleParticle, self).__init__()  
        self.visualization = None
        self.numParticles = numParticles
        self.phi = phi
        self.stateBounds = stateBounds #[pMin, pMax, vMin, vMax]
        self.dwellTime = dwellTime
        self.maxSteps = maxSteps
        
        #action space is 4 discrete values (right, up, left, down)
        self.action_space = spaces.Discrete(4)
        
        #observation space is 4 x numParticles
        self.observation_space = spaces.Box(
                                 np.array([stateBounds[0], stateBounds[2]]), #LB
                                 np.array([stateBounds[1], stateBounds[3]]), #UB
                                 dtype=np.float32)
        


    def reset(self):
        # Reset the state of the environment to an initial state
        
        #### set the state randomly within +- 40% of the bounds
        #self.states = np.random.default_rng().uniform(low=-1.0, high=1.0, size=self.numParticles*4)
        #self.states[0:2] = (self.states[0:2] + 1) / 2 * (self.stateBounds[1] - self.stateBounds[0]) + self.stateBounds[0]
        #self.states[0:2] *= 0.4 #start within 40% of the origin
         
        #### set the state to a fixed point
        self.states = np.array([self.stateBounds[0], self.stateBounds[0], self.stateBounds[3], self.stateBounds[3]])
        self.states[0:2] *= 0.4; #scale position to be 40% of max distance
        
        
        # Convert to a 32 bit float to play nice with the pytorch tensors
        self.states = self.states.astype('float32')
        self.currentStep = 0
        if self.visualization != None:
            self.visualization.reset()


    def _next_observation(self):
        #observe the current state of the particles
        obs = self.states;
        return obs

    #update particle states based on the current action & dynamics
    def _take_action(self, action):
        #calculate the velocity (v=vmax always)
        vMax = self.stateBounds[3];
        #determine how the agent would move if the offset was 0
        v = np.array([ [vMax], [0] ]) #assume to the right (action = 0)
        if action == 1: #up
            v = np.array([ [0], [vMax] ])
        if action == 2: #left
            v = np.array([ [-vMax], [0] ])
        if action == 3: #down
            v = np.array([ [0], [-vMax] ])

            
        #calculate the rotation of the particle
        c, s = np.cos(self.phi), np.sin(self.phi)
        R    = np.array( [[c, -s], [s, c]] )
        #rotate u by angle phi (particle offset)
        v = np.matmul(R, v)
        #do the dynamics
        position = self.states[:2] + np.transpose(v * self.dwellTime)
        self.states = np.append(position, v)        
        self.states = self.states.astype('float32')
        


    def step(self, action):
        # Execute one time step within the environment
        self.old_states = self.states
        self._take_action(action)
        self.currentStep += 1 #we just advanced by one dwellTime
        
        #check if we have reached our timeout or otherwise ended
        done = False #todo: check if we come within distance of origin
        if self.currentStep >= self.maxSteps:
            done = True
    
        ### Calculate the Reward ###
        '''Reward is the minimum distance between the origin and
        the particle as it moves between the initial and final states'''
        x0 = self.old_states[0]
        xf = self.states[0]
        y0 = self.old_states[1]
        yf = self.states[1]
        
        #calculate slope from dx and dy
        dx = xf - x0
        dy = yf - y0
        
        #first, we can assume cost = distance from final position to origin
        dist = np.linalg.norm(self.states[0:2])
        #check if we are passing by the origin (so we could be closer)
        if xf*x0 < 0 or yf*y0 < 0:
            if dx == 0: #moving only vertically
                dist = abs(x0)
            if dy == 0: #moving only horizontally
                dist= abs(y0)
            if dx != 0 and dy != 0: #moving at an angle
                m = dy / dx
                x = (m*m*x0 + m*y0) / (m*m + 1)
                y = m*(x - x0) + y0
                dist = math.min(dist, math.sqrt(x*x + y*y))
        
        #reward the agent by -1 * dist from origin
        reward = -dist
        
        
        #penalize the agent by 1,000 if it exits the state bounds
        if xf < self.stateBounds[0] or xf > self.stateBounds[1] or yf < self.stateBounds[0] or yf > self.stateBounds[1]:
                reward -= 1000;
        
        #finally, convert to float32 to play nice with pytorch
        reward = reward.astype('float32')
        
        #finish if we are within 0.01 microns of the goal
        done = done or (dist < 0.01)
                
        newState = self.states
    
    
        #return state, reward, done, next state
        return self.states, reward, done, newState 
    

    def _render_to_file(self, filename='data'):
        # Append the current run to filename (default data.csv)
        file = open(filename + '.csv', 'a+')
        
        file.write(self.dwellTime * self.step, ",", self.states)
        
        file.close()


    def render(self, mode='live', title=None, **kwargs):
        # Render the environment to the screen or a file
        if mode == 'file':
            #render to file (default to data.csv)
            self._render_to_file(kwargs.get('filename', 'data'))
        elif mode == 'live':
            if self.visualization == None:
                self.visualization = ParticlePlot(title,  window_size=self.stateBounds)
                
            #scale rendering to the state bounds instead of [-1, 1] in each dim
            self.visualization.render(self.states)
        
        
    def close(self):
        super().close()
        
        if self.visualization != None:
            self.visualization.close()
        
