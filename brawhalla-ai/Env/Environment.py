# a warp-up for Sensor, Preprocessor, & Actuator
from .Actuator import Actuator
from .Sensor import Sensor
from .Preprocessor import Preprocessor
import numpy as np
import time
import cv2

class Environment:
    def __init__(self):
        self.sensor = Sensor()
        self.preprocessor = Preprocessor()
        self.actuator = Actuator()
        self.action_space = self.actuator.action_space
    def get_state(self):
        # get current state
        return self.preprocessor.preprocess(self.sensor.screen_record())

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
        print('reward: {}' . format(reward))
        obs = self.preprocessor.preprocess(obs) # H x W
        return obs, reward, False



"""def wait(T):
    for i in list(range(T))[::-1]:
        print(i+1)
        time.sleep(1)
wait(4)
env = Environment()
obs = env.reset()
while True:
    last_time = time.time()
    print(np.array(obs).shape)
    cv2.imshow('window', obs)

    obs, reward = env.step(0) #0.03 s
    print('loop took {0} seconds' . format(time.time()-last_time))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break"""
