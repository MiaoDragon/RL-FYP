import importlib.util
from Human import Human
from ai_random import Agent as AgentR
from ai_naive import Agent as AgentN
class Admin:
    # this controls the whole game loop
    def start(self, T=200, R=2000, wflag=False, alpha=0.1):
        # T is the maximum # of episodes
        # R is the maximum # of rounds in each episode
        stat = 0.
        if wflag:
            f = open('stat-alpha{0}.txt' . format(alpha), 'w')
        print('#####################################################')
        print('Welcome to rock-paper-scissor Game!')
        print('*****************************************************')
        pid = [-1,-1]
        p = [None, None]
        p[0] = AgentR(alpha)
        p[1] = AgentN(alpha)

#        for i in range(2):
#            print('--Choose player {0}--' . format(1))
#            print('Options:')
#            print('1. human')
#            print('2. AI (please input the AI file, it needs to have model name as Agent.)')
#            print('enter 1 or 2 to choose.')
#            while True:
#                try:
#                    pid[i] = int(input())
#                    break
#                except ValueError:
#                    print("Not a number. Try again.")
#            if pid[i] == 2:
#                print('Please input the file location:')
#                while True:
#                    path = str(input())
#                    try:
#                        spec = importlib.util.spec_from_file_location('Agent', path)
#                        agent = importlib.util.module_from_spec(spec)
#                        spec.loader.exec_module(agent)
#                        p[i] = agent.Agent()
#                        break
#                    except IOError:
#                        print('path is not correct. Try again.')
#            else:
#                p[i] = Human()

        # start the game flow
        actions = ['rock', 'paper', 'scissor']
        Tscores = [0,0]
        for t in range(T):
            print('----------------------------------')
            print('{0}th episode.' . format(t+1))
            Rscores = [0,0]
            p[0].reset((0,0))
            p[1].reset((0,0))
            for r in range(R):
                print('------{0}th round.------' . format(r+1))
                action0 = p[0].decide()
                action1 = p[1].decide()
                print('action: {0},{1}'. format(action0, action1))
                print('result: ')
                print('player0: ' + actions[action0-1])
                print('player1: ' + actions[action1-1])
                def isLarger(a1, a2, p1, p2):
                    # compare the value of a1, a2. print if p1 wins
                    nonlocal Rscores
                    if a1 > a2 and not (a1==3 and a2==1):
                        print('player{0} wins!' . format(p1))
                        Rscores[p1] += 1
                        return True
                    if a1 == 1 and a2 == 3:
                        print('player{0} wins!' . format(p1))
                        Rscores[p1] += 1
                        return True
                    return False
                if action0 == action1:
                    print('draw.')
                    p[0].observe((action0, action1), 0)
                    p[1].observe((action1, action0), 0)
                else:
                    if isLarger(action0, action1, 0, 1):
                        p[0].observe((action0, action1), 1)
                        p[1].observe((action1, action0), -1)
                    if isLarger(action1, action0, 1, 0):
                        p[1].observe((action1, action0), 1)
                        p[0].observe((action0, action1), -1)
            print('****End of this round.****')
            print('result: {0}:{1}' . format(Rscores[0], Rscores[1]))
            f.write('{0}\n' . format(Rscores[1]-Rscores[0]))
            stat += Rscores[1]-Rscores[0]
            if Rscores[0] > Rscores[1]:
                Tscores[0] += 1
            elif Rscores[1] > Rscores[0]:
                Tscores[1] += 1
        stat = stat / (T*R)
        f.write('average: {0}\n' . format(stat))
        f.close()
        p[0].term()
        p[1].term()



admin = Admin()
alphas = [0.2, 0.1, 0.05, 0.025, 0.01]
for alpha in alphas:
    admin.start(wflag=True, alpha=alpha)
