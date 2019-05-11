from agent_a3c import Agent
import threading
import gym
import numpy as np
import time
from image_processor import ImageProcessor

class Environment(threading.Thread):
    stop_signal = False

    def __init__(self, brain, eps_start, eps_end, eps_steps, thread_delay, env_name, gamma, gamma_n, n_step_return, render=False):
        threading.Thread.__init__(self)

        self.render = render
        self.env = gym.make(env_name)
        self.brain = brain
        self.agent = Agent(eps_start, eps_end, eps_steps, self.env.action_space.n, gamma, gamma_n, n_step_return, self.brain)
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.thread_delay = thread_delay
        self.dims = (84,84)
        self.current_state = np.zeros((4, self.dims[1], self.dims[0]), dtype=np.uint8)
        self.image_processor = ImageProcessor()
        self.fire = 1 #fix to make generic
        self.true_done = False
        self.lives = 0
        self.total_ep_reward = 0

    def addToCurrentState(self, state):
        self.current_state = np.roll(self.current_state, 1, axis=0)
        self.current_state[0] = self.image_processor.preprocess(state, self.dims, False)

    def getCurrentState(self):
        return self.current_state
        # return np.transpose(self.current_state, (1, 2, 0))

    def runEpisode(self):
        if self.true_done or self.lives == 0:
            self.env.reset()
            self.lives = self.env.unwrapped.ale.lives()
            self.total_ep_reward = 0

        sta_1, _, _, _ = self.env.step(self.fire)
        self.addToCurrentState(sta_1)
        for i in range(3):
            sta, _, _, _ = self.env.step(self.env.action_space.sample())
            self.addToCurrentState(sta)
        R = 0
        while True:
            time.sleep(self.thread_delay)  # yield
            current = self.getCurrentState()
            if self.render: self.env.render()
            s = current
            if self.env.unwrapped.ale.lives() < self.lives:
                a = self.fire
            else:
                a = self.agent.act(current)
            sta, r, done, info = self.env.step(a)
            self.true_done = done
            self.addToCurrentState(sta)
            if self.env.unwrapped.ale.lives() < self.lives and self.brain.total_episodes > 4000:
                done = True
            self.agent.train(s, a, r, done)
            R += r
            self.lives = self.env.unwrapped.ale.lives()
            if done:
                self.total_ep_reward += R
                R=0
            if self.true_done or self.stop_signal:
                self.brain.add_rewards(self.total_ep_reward)
                break
            
                
        if self.total_ep_reward > 11 and self.true_done:
            print("Total R:", self.total_ep_reward)

    def run(self):
        while not self.stop_signal:
            self.runEpisode()

    def stop(self):
        self.stop_signal = True