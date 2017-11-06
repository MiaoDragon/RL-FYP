# ref: http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
import matplotlib
matplotlib.use('Agg') # turn off showing, from https://stackoverflow.com/questions/29125228/python-matplotlib-save-graph-without-showing
import matplotlib.pyplot as plt
import torch
class Plotter:
    def __init__(self, folder='DQN/plot'):
        self.train_rewards = []
        self.errors = []
        self.eval_rewards = []
        self.folder = folder
        self.SAVE_FREQ = 100
        plt.ion()
        #plt.ioff()
        plt.figure()

    def plot_errors(self, error):
        self.errors.append(error)
        plt.figure(1)
        plt.clf()
        error_t = torch.FloatTensor(self.errors)
        plt.title('Training...')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.plot(error_t.numpy())
        # average of 100 frames
        if len(error_t) >= 100:
            means = error_t.unfold(0,100,1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        plt.pause(0.001)

    def plot_train_rewards(self,R):
        self.train_rewards.append(R)
        plt.figure(2)
        plt.clf()
        train_reward_t = torch.FloatTensor(self.train_rewards)
        plt.title('Training...')
        plt.xlabel('Epoch')
        plt.ylabel('Train_rewards')
        plt.plot(train_reward_t.numpy())
        # average of 100 frames
        if len(train_reward_t) >= 100:
            means = train_reward_t.unfold(0,100,1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        plt.pause(0.001)
        if len(train_reward_t) % self.SAVE_FREQ == 0:
            # save every 100 iterations
            plt.savefig(self.folder+'/DQN-cartpole-train_reward-100f', dpi=300)

    def plot_eval_rewards(self, R):
        self.eval_rewards.append(R)
        plt.figure(3)
        plt.clf()
        eval_reward_t = torch.FloatTensor(self.eval_rewards)
        plt.title('Training...')
        plt.xlabel('Epoch')
        plt.ylabel('Eval_rewards')
        plt.plot(eval_reward_t.numpy())
        # average of 100 frames
        if len(eval_reward_t) >= 100:
            means = eval_reward_t.unfold(0,100,1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        plt.pause(0.001)

    def terminate(self):
        plt.figure(1)
        plt.savefig(self.folder+'/DQN-boxing-100-scaled-error-100f', dpi=300)
        plt.figure(2)
        plt.savefig(self.folder+'/DQN-boxing-100-scaled-train_reward-100f', dpi=300)
        plt.figure(3)
        plt.savefig(self.folder+'/DQN-boxing-100-scaled-eval_reward-100f', dpi=300)
        plt.ioff()
        plt.show()
