import numpy as np
class Env:
    def __init__(self, states, t_states,num_state, num_action=2):
        # states: [state]
        # state: [action]
        # action: [{state, reward, prob.}]
        self.reward = [[[0. for i in range(num_state)] for j in range(num_action)] for k in range(num_state)]
        self.prob = [[[0. for i in range(num_state)] for j in range(num_action)] for k in range(num_state)]
        self.num_state = num_state
        self.num_action = num_action
        self.t_states = t_states # terminating states
        for i in range(num_state):
            for j in range(num_action):
                for action in states[i][j]:
                    self.reward[i][j][action['state']]=action['reward']
                    self.prob[i][j][action['state']]=action['prob']
    def reset(self):
        self.state = 0
        return self.state
    def step(self,action):
        if self.state in self.t_states:
            print('already terminated')
            return self.state, 0, True
        elements = [i for i in range(self.num_state)]
        probs = self.prob[self.state][action]
        next_state = np.random.choice(elements, p=probs)
        reward = self.reward[self.state][action][next_state]
        self.state = next_state
        term = False
        if next_state in self.t_states:
            term = True
        return self.state, reward, term
