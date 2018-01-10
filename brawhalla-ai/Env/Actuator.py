import keyboard
import time
import random
def wait(T):
    for i in list(range(T))[::-1]:
        print(i+1)
        time.sleep(1)

class ActionSpace:
    def __init__(self, cap):
        self.capacity = cap
        self.actions = [i for i in range(self.capacity)]
    def sample(self):
        # random sample
        return random.sample(self.actions, 1)[0]
    def __len__(self):
        return self.capacity
    def __str__(self):
        return str(self.actions)

class Actuator:
# problem specific: Instant Action + Charged Action + Mixed
    def __init__(self):
        self.actions = ['W','J','L', 'H',  # Instant Actions
                        'A','S','D','K','AS','SD','AK','DK','SK','WK',
                        'AH','AWH','WH','WDH','DH','SDH','SH','ASH',     # Charged Actions
                        'AW','DW','AJ','SJ','DJ',
                        'AL','ASL','SL','SDL','DL','DWL','WL','AWL'      # Mixed Actions
                        ]
        self.keys = ['A','S','D','W','H','J','K','L']
        self.instant_num = 4
        self.charged_num = 18
        self.mixed_num = 13
        self.action_space = ActionSpace(self.instant_num + self.charged_num + self.mixed_num)

    def release_except(self, keys):
        for k in self.keys:
            if k not in keys and keyboard.is_pressed(k):
                #print('releasing {}' . format(k))
                keyboard.release(k)

    def actuate(self, signal):
        if signal < self.instant_num:
            # release all others, and press_and_release
            self.release_except('')
            keyboard.press_and_release(self.actions[signal])
        elif signal < self.instant_num+self.charged_num:
            # release all other keys except the direction keys, and press
            # to make the action more smooth
            actions = self.actions[signal]
            if len(actions) > 1:
                self.release_except(actions[:-1])
            else:
                self.release_except(actions)
            for k in actions:
                #print(k)
                keyboard.press(k)
        else:
            # release all other keys except the direction keys, and press
            # to make the action more smooth
            actions = self.actions[signal]
            self.release_except(actions[:-1])
            for k in actions[:-1]:
                #print(k)
                keyboard.press(k)
            keyboard.press_and_release(actions[-1])
