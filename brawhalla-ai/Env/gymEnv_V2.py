import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import gym                      # for testing

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
class myGym:
    # a wrapper for gym env for testing
    def __init__(self):
        self.env = gym.make('CartPole-v0').unwrapped
        self.screen_width = 600
        self.action_space = self.env.action_space
        self.show = 0
        self.prev_screen = None
    def reset(self):
        obs = self.env.reset()
        self.prev_screen = self.get_screen()
    def step(self, action):
        self.prev_screen = self.get_screen()  # remember the previous screen
        obs, r, done, _ = self.env.step(action)
        if done:
            self.env.reset()  # here prev_screen is also updated
        return self.get_state(), r, done

    def get_cart_location(self):

        world_width = self.env.x_threshold * 2
        scale = self.screen_width/world_width
        return int(self.env.state[0] * scale + self.screen_width/2.0)

    def get_screen(self):
        #print('inside get_state')
        screen = self.env.render(mode='rgb_array').transpose( (2,0,1) )
        #print('got the screen...')
        # strip off the top and bottom of the screen
        screen = screen[:, 160:320]
        #view_width = 320
        #cart_location = self.get_cart_location()
        #if cart_location < view_width // 2:
        #    slice_range = slice(view_width)
        #elif cart_location > (self.screen_width-view_width // 2):
        #    slice_range = slice(-view_width, None)
        #else:
        #    slice_range = slice(cart_location - view_width // 2,
        #                        cart_location + view_width // 2)
        #screen = screen[:, :, slice_range]
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.0

        screen = screen.transpose((1,2,0))
        screen = cv2.resize(screen, dsize=(80,40), interpolation = cv2.INTER_CUBIC)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        res = screen
        #cv2.imshow('image', screen)
        #print(screen.shape)
        #res = screen.transpose((2,0,1))
        return res

    def get_state(self):
        self.show += 1
        screen = self.get_screen()
        res = 1. - np.array([self.prev_screen, screen])  # inverting the color

        #print(res.shape)
        #res = screen - self.prev_screen
        #cv2.imshow('image', 1-screen)
        #cv2.imshow('image-2', 1-self.prev_screen)
        return res
