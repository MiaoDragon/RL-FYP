# a warp-up for Sensor, Preprocessor, & Actuator
from .Actuator import Actuator
from .Sensor import Sensor
from .Preprocessor import Preprocessor
import numpy as np
import time
import cv2
import random

class Environment:
    #def __init__(self):
    #    pass
        #self.sensor = Sensor()
        #self.preprocessor = Preprocessor()
        #self.actuator = Actuator()
        #self.action_space = self.actuator.action_space
    def set_all(self):
        self.sensor = Sensor()
        self.preprocessor = Preprocessor()
        self.actuator = Actuator()
        self.action_space = self.actuator.action_space
    def get_state(self):
        # get current state
        return np.array([self.preprocessor.preprocess(self.sensor.screen_record())])

    def reset(self):
        # get current obs
        return self.sensor.screen_record()

    def step(self, signal):
        # return obs, reward, done
        # achieve an action
        self.actuator.actuate(signal)
        # TODO: may need sleep here
        obs = self.sensor.screen_record()
        reward = self.preprocessor.reward_cal(obs)
        if reward != 0:
            print('reward: {}' . format(reward))
        else:
            reward = 0.01 # alive
        obs = np.array([self.preprocessor.preprocess(obs)]) # H x W
        flag = False
        if reward == -2.0:
            flag = True
        return obs, reward, flag



def wait(T):
    for i in list(range(T))[::-1]:
        print(i+1)
        time.sleep(1)

"""
wait(4)
env = Environment()
env.set_all()
obs = env.get_state()
while True:
    last_time = time.time()
    print(np.array(obs).shape)
    cv2.imshow('window', obs.transpose((1,2,0)))

    obs, reward, done = env.step(random.sample(range(35),1)[0]) #0.03 s
    #print('loop took {0} seconds' . format(time.time()-last_time))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
"""
