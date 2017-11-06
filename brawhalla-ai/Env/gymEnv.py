import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import gym                      # for testing
class myGym:
    # a wrapper for gym env for testing
    def __init__(self):
        self.env = gym.make('CartPole-v0').unwrapped
        self.env.reset()
        self.screen_width = 600
        self.action_space = self.env.action_space
    def reset(self):
        obs = self.env.reset()
        return self.get_state()
    def step(self, action):
        obs, r, done, _ = self.env.step(action)
        if done:
            obs = self.env.reset()
        return self.get_state(), r, done

    def get_cart_location(self):

        world_width = self.env.x_threshold * 2
        scale = self.screen_width/world_width
        return int(self.env.state[0] * scale + self.screen_width/2.0)

    def get_state(self):
        #print('inside get_state')
        screen = self.env.render(mode='rgb_array').transpose( (2,0,1) )
        #print('got the screen...')
        # strip off the top and bottom of the screen
        screen = screen[:, 160:320]
        view_width = 320
        cart_location = self.get_cart_location()
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (self.screen_width-view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        screen = screen[:, :, slice_range]
        screen = np.ascontiguousarray(screen, dtype=np.float32)
        screen = screen.transpose((1,2,0))
        screen = cv2.resize(screen, (40,80), interpolation = cv2.INTER_CUBIC)
        res = screen.transpose((2,0,1))
        return res
        #screen = torch.from_numpy(screen)  # convert to torch tensor
        #return resize(screen).unsqueeze(0).type(Tensor)
