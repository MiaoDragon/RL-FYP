from AgentBase import AgentBase
class Human(AgentBase):
    def decide(self):
        print('===Please input your choice of action:===')
        print('1. rock')
        print('2. paper')
        print('3. scissor')
        print('Enter 1, 2 or 3 to choose.')
        while True:
            try:
                action = int(input())
                if not (action >= 1 and action <= 3):
                    raise ValueError
                break
            except ValueError:
                print("Not a number. Try again.")
        return action
    def observe(self, observaion, reward):
        pass
    def reset(self, observation):
        pass
