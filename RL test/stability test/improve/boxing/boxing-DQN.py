# ref: https://github.com/dennybritz/reinforcement-learning/blob/master/DQN/Deep%20Q%20Learning.ipynb
import tensorflow as tf
import gym
import numpy as np
import random
import cv2
import time
class QNet:
    def __init__(self):
        # build model
        # obs->action q value
        self.learning_rate = 0.00025
        self.batchsize = 32

        self.input_layer = tf.placeholder(shape=(None, 84, 84, 4), dtype=tf.float32)
        self.conv1 = tf.layers.conv2d(
            inputs=self.input_layer,
            filters=32,
            kernel_size=[8,8],
            padding='valid',
            strides=(4,4),
            activation=tf.nn.relu
        )
        #self.pool1 = tf.layers.max_pooling2d()
        self.conv2 = tf.layers.conv2d(
            inputs=self.conv1,
            filters=64,
            kernel_size=[4,4],
            padding='valid',
            strides=(2,2),
            activation=tf.nn.relu
        )
        #self.pool2
        self.conv3 = tf.layers.conv2d(
            inputs=self.conv2,
            filters=64,
            kernel_size=[3,3],
            padding='valid',
            strides=(1,1),
            activation=tf.nn.relu
        )
        #outsize: 7x7x64
        self.conv3_flat = tf.reshape(self.conv3, [-1, 7*7*64])
        self.dense = tf.layers.dense(
            inputs=self.conv3_flat,
            units=512,
            activation=tf.nn.relu
        )
        #dropout
        w_out = tf.Variable(tf.truncated_normal([512,18],stddev=0.01))
        b_out = tf.Variable(tf.constant(0.01, shape=[18]))
        self.Qs = tf.matmul(self.dense,w_out)+b_out
        #self.Qs = tf.layers.dense(
        #    inputs=self.dense,
        #    units=18,
        #)
        self.actions = tf.placeholder(shape=[None, 18],dtype=tf.float32)
        #self.one_hot_actions = tf.one_hot(indices=self.actions,depth=18,on_value=1.0,off_value=0.0,dtype=tf.float32)
        #self.one_hot_actions = tf.reshape(self.one_hot_actions, shape=(-1, 18))
        #self.Q = tf.matmul(tf.multiply(self.Qs,self.one_hot_actions),tf.ones(shape=(18,1),dtype=tf.float32))
        self.Q = tf.reduce_sum(tf.multiply(self.Qs, self.actions), axis=1)
        #self.Q = tf.reshape(self.Q, shape=[-1])
        # after multiply elementwisely, the size is m * 18, then matrix multiply 1, then result is m * 1
        self.nextQ = tf.placeholder(shape=[None],dtype=tf.float32)

        self.loss = tf.reduce_mean( tf.square(self.nextQ-self.Q) )
        self.average_loss = tf.reduce_mean( tf.square(self.nextQ-self.Q) )
        #self.trainer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        # original paper: use RMSProp
        self.trainer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,momentum=0.95)
        self.updateModel = self.trainer.minimize(self.loss)


    def eval(self, sess, s):
        return sess.run(self.Qs[0], { self.input_layer: np.array([s]) })


    def train(self, sess, exps, targetNet):
        # minibatch
        indexs = random.sample([i for i in range(len(exps[0]))], self.batchsize)
        newexps = [np.array([exps[i][j] for j in indexs]) for i in range(len(exps))]

        nextQs = getTarget(sess, targetNet, newexps[2], newexps[0])
        Qs, avg_loss, _ = sess.run([self.Q, self.average_loss,self.updateModel], {self.input_layer: newexps[0], self.actions: newexps[1],
                                    self.nextQ: nextQs})
        return Qs, avg_loss

def restore(sess, saver, save_dir, filename=None):
    if filename == None:
        # auto restore
        latest_checkpoint = tf.train.latest_checkpoint(save_dir)
        if latest_checkpoint:
            print('Loading model checkpoint {}...\n' . format(latest_checkpoint))
            saver.restore(sess,latest_checkpoint)
            return True
        else:
            return False
    saver.restore(sess,save_dir+filename)

def preprocess(image):
    # convert to numpy
    image = np.array(image)
    # the original paper preprocesses the previous image together with the current one to remove flickering
    # in this version, simply preprocess the current image
    # extract the Y channel, and rescale to 84 x 84
    #image *= 1./255   # normalize
    #print(image)
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y, _, _ = cv2.split(image)
    # scale
    return cv2.resize(y, (84,84), interpolation = cv2.INTER_CUBIC)


def getTarget(sess, targetNet, r, s):
    # find max q value for s
    gamma = 0.99  # we assume it is continual RL problem, as the evaluation duration is large
    # old: get target one by one
    #Qs = [targetNet.eval(sess, s, i) for i in range(18)]
    # new: get target for a batch of states
    Qs = sess.run(targetNet.Qs, { targetNet.input_layer: np.array(s) })
    return r+gamma*Qs.max(axis=1)

def q_learning(env, num_trainepi=5000, num_evalepi=500, epilen=18000):
    #exp_size = 10000
    exp_size = 10000
    transfer_time = 100
    epislon = 1. # decay as train increase
    epislon_decay = 0.8 # decay per episode
# original paper: epislon init, epislon final, num_frames to get final epislon
    train_freq = 1. # train each # frame
    stack_size = 4
    memory_stack = []
# original paper: keep action fixed for a series of time
# original paper: random start
    exp_replay = [[],[],[],[]] #state, action, reward, next_state
    qNet = QNet()
    targetNet = QNet()
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    save_dir = 'models/'
    filename = 'boxing-DQN_model.ckpt'
    save_path = save_dir + filename

    one_hot_actions = np.eye(18)

    def update(sess, exp_replay):
        return qNet.train(sess, exp_replay, targetNet)

    def memory_stack_push(obs, memory_stack):
        # push the current state
        state = preprocess(obs)
        if len(memory_stack) == stack_size:
            memory_stack.pop(0)
        memory_stack.append(state)
        if len(memory_stack) < stack_size:
            # padding with the current memory
            memory_stack += [state for i in range(stack_size-len(memory_stack))]
    def memory_to_np(memory_stack):
        return np.swapaxes(np.swapaxes(np.array(memory_stack),0,1),1,2)

    def eval_one_epi(env,renderflag=False):
        R = 0.
        memory_stack = []
        obs = env.reset()
        for _ in range(epilen):
            if renderflag:
                env.render()
            memory_stack_push(obs, memory_stack)
            Qs = qNet.eval(sess,memory_to_np(memory_stack))
            action = np.argmax(Qs)
            obs, r, done, info = env.step(action)
            R += r
            if done:
                break
        print('total reward: {}' . format(R))
        return R

    # define copy operation
    params = tf.trainable_variables()
    qlen = len(params) // 2
    copy_ops = [params[i+qlen].assign(params[i].value()) \
                for i in range(qlen)]

    # init exp_replay by random sampling
    obs = env.reset()
    memory_stack = []
    memory_stack_push(obs, memory_stack)
    while len(exp_replay[0]) < exp_size:
        #env.render()
        action = env.action_space.sample()
        exp_replay[0].append(memory_to_np(memory_stack))
        obs, r, done, _ = env.step(action)
        exp_replay[1].append(one_hot_actions[action])
        exp_replay[2].append(r)
        memory_stack_push(obs, memory_stack)
        exp_replay[3].append(memory_to_np(memory_stack))
        if done:
            memory_stack = []
            obs = env.reset()
            memory_stack_push(obs, memory_stack)


    with tf.Session() as sess:
        # TODO: add restore flag
        sess.run(init)
        try:
            restore(sess, saver, save_dir, filename)
        except:
            restore(sess, saver, save_dir)
        # pretrain
        print('Pretraining...')
        _, avg_loss = update(sess, exp_replay)
        epi = 0

        #train
        while epi < num_trainepi:
            # train episode
            print('@@@ Training episode: {}' . format(epi))
            memory_stack = [] # restart, clear memory
            obs = env.reset()
            memory_stack_push(obs, memory_stack)
            step = 0
            # store the weights
            if epi % 100 == 0:
                saver.save(sess, save_path)
            last_time = time.time()
            for _ in range(epilen):
                step += 1
                #env.render()
                #state = preprocess(obs)
                # select the action based on behavior policy
                if np.random.rand(1) < epislon:
                    # random choose
                    action = env.action_space.sample()
                else:
                    Qs = qNet.eval(sess,memory_to_np(memory_stack))
                    action = np.argmax(Qs)

                exp_replay[0].pop(0)
                exp_replay[1].pop(0)
                exp_replay[2].pop(0)
                exp_replay[3].pop(0)
                exp_replay[0].append(memory_to_np(memory_stack))
                exp_replay[1].append(one_hot_actions[action])
                # step
                obs, r, done, info = env.step(action)
                exp_replay[2].append(r)
                memory_stack_push(obs, memory_stack)
                exp_replay[3].append(memory_to_np(memory_stack))
                # train with freq
                if np.random.rand(1) < train_freq:
                    Qs, avg_loss = update(sess, exp_replay)
                    if step % 100 == 0:
                        print('Q values: {}' . format(Qs))
                        print('average loss: {}' . format(avg_loss))
                # transfer when it is time
                if step % transfer_time == 0:
                    sess.run(copy_ops)
                if done:
                    break
            print('##training took {0} seconds' . format(time.time()-last_time))
            last_time = time.time()
            epislon = epislon * epislon_decay
            epi += 1
            # evaluate one episode
            print('evaluate for one episode...')
            if epi % 5 == 0:
                eval_one_epi(env,False)
            else:
                eval_one_epi(env,False)
            print('##evaluation took {0} seconds' . format(time.time()-last_time))

        # evaluate
        epi = 0
        while epi < num_evalepi:
            # use actual policy
            print('@@@ Evaluation episode: {}' . format(epi))
            memory_stack = []
            obs = env.reset()
            R = 0.
            for _ in range(epilen):
                #if epi % 5 == 0: # every 100 timess
                #    env.render()
                memory_stack_push(obs, memory_stack)
                Qs = qNet.eval(sess,memory_to_np(memory_stack))
                #Qs = [qNet.eval(sess,memory_to_np(memory_stack),i) for i in range(18)]
                action = np.argmax(Qs)
                obs, r, done, info = env.step(action)
                R += r
                if done:
                    break
            print('total reward: {}' . format(R))
            epi += 1



def start():
    env = gym.make('Boxing-v0')
    q_learning(env, num_trainepi=10, num_evalepi=10,epilen=900)

start()
