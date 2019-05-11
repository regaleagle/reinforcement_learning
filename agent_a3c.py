import random
import numpy as np
from collections import deque

class Agent:
    def __init__(self, eps_start, eps_end, eps_steps, num_actions, gamma, gamma_n, n_step_return, brain):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps
        self.num_actions = num_actions
        self.gamma = gamma
        self.gamma_n = gamma_n
        self.n_step_return = n_step_return
        self.memory = (deque(maxlen=self.n_step_return), deque(maxlen=self.n_step_return), deque(maxlen=self.n_step_return), deque(maxlen=self.n_step_return))  # s, a, r, s'  # used for n_step return
        # self.R = 0.
        self.brain = brain
        self.frames = 0

    def getEpsilon(self):
        if (self.frames >= self.eps_steps):
            return self.eps_end
        else:
            return self.eps_start + self.frames * (self.eps_end - self.eps_start) / self.eps_steps  # linearly interpolate

    def act(self, s):
        #eps = self.getEpsilon()
        self.frames += 1
        s = np.array([s])
        p = self.brain.predict_p(s)[0]
        if self.frames < 10000 or np.argmax(p) > 0.8:
            a = np.random.choice(self.num_actions, p=p)
        else:
            a = np.argmax(p)
        return a
        # if random.random() < eps:
        #     a = np.random.choice(self.num_actions, p=p)
        #     return a
        #
        # else:
        #     a = np.argmax(p)
        #     # if p[3] > .5 or p[2] > .5:
        #     #     print(p)
        #     return a

    def train(self, s, a, r, done):
        # def get_sample(memory, n):
        #     s, a, _, _ = memory[0]
        #     _, _, _, s_ = memory[n - 1]
        #
        #     return s, a, self.R, s_

        # a_cats = np.zeros(self.num_actions)  # turn action into one-hot representation
        # a_cats[a] = 1

        self.memory[0].append(s)
        self.memory[1].append(a)
        self.memory[2].append(r)
        self.memory[3].append(done)
        if len(self.memory) > 2000:
            print(len(self.memory))

        # self.R = (self.R + r * self.gamma_n) / self.gamma

        if (done and len(self.memory[0]) > 0) or len(self.memory[0]) >= self.n_step_return:
                s_, a_, r_, d_ = self.memory
                self.brain.train_push(np.array(s_), np.array(a_), np.array(r_), np.array(d_))
                #if done:
                self.memory = (deque(maxlen=self.n_step_return), deque(maxlen=self.n_step_return), deque(maxlen=self.n_step_return), deque(maxlen=self.n_step_return))
        # if len(self.memory) >= self.n_step_return:
        #     s, a, r, s_ = get_sample(self.memory, self.n_step_return)
        #     self.brain.train_push(s, a, r, s_)
        #
        #     self.R = self.R - self.memory[0][2]
        #     self.memory.pop(0)

            # possible edge case - if an episode ends in <N steps, the computation is incorrect