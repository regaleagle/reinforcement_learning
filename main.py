# import gym
# from gym import spaces
# import random
# import numpy as np
# import time
# from agent import Agent
# from image_processor import ImageProcessor
# import tensorflow as tf
# import os
# from dqn import Dqn
# from keras import backend as K
#
# #np.set_printoptions(threshold=sys.maxsize)
# stored_model = "the_model_ddqn_6"
# env = gym.make('BreakoutDeterministic-v4')
# image_processor = ImageProcessor()
# replay_length = 200000
# dims = (84,84)
# current_state = np.zeros((4, dims[1], dims[0]), dtype=np.uint8)
#
# # PATH = "output/"                 # Gifs and checkpoints will be saved here
# # SUMMARIES = "summaries"          # logdir for tensorboard
# # RUNID = 'run_2'
# # os.makedirs(PATH, exist_ok=True)
# # os.makedirs(os.path.join(SUMMARIES, RUNID), exist_ok=True)
# # SUMM_WRITER = tf.summary.FileWriter(os.path.join(SUMMARIES, RUNID))
# #
# # # main DQN and target DQN networks:
# with tf.variable_scope('mainDQN'):
#     model = Dqn(current_state.shape, env.action_space.n)
# with tf.variable_scope('targetDQN'):
#     target_model = Dqn(current_state.shape, env.action_space.n)
# #
# # init = tf.global_variables_initializer()
# #
# # MAIN_DQN_VARS = tf.trainable_variables(scope='mainDQN')
# # # TARGET_DQN_VARS = tf.trainable_variables(scope='targetDQN')
#
#
#
# agent = Agent(current_state.shape, env.action_space.n, model, target_model, replay_length)
#
# print(env.unwrapped.get_action_meanings())
# #openCV inverts order of dims
#
# print(current_state.shape)
# print(env.action_space)
# print(env.action_space.n)
#
# ####PREFILL
#
#
# state = env.reset()
# start = time.time()
#
#
# try:
#     agent.memory_current = np.load('numpy_prefill_memory_current.npy')
#     agent.memory_reward = np.load('numpy_prefill_memory_reward.npy')
#     agent.memory_action = np.load('numpy_prefill_memory_action.npy')
#     agent.memory_done = np.load('numpy_prefill_memory_done.npy')
#     agent.memory_next = np.load('numpy_prefill_memory_next.npy')
# except IOError:
#     prefill_counter = 0
#     print("start prefill")
#     while prefill_counter < replay_length:
#         done = False
#         env.reset()
#         print("counter = ", prefill_counter)
#         while not done:
#             previous_state = current_state
#             action = env.action_space.sample()
#             state, reward, done, _ = env.step(action)
#             img = image_processor.preprocess(state, dims)
#             current_state = np.roll(current_state, 1, axis=0)
#             current_state[0] = img
#             agent.addToMemory(previous_state, action, reward, current_state, done, prefill_counter)
#             prefill_counter += 1
#
#         end = time.time()
#
#     print("End of prefill")
#     print(end - start)
#     np.save('numpy_prefill_memory_current', agent.memory_current)
#     np.save('numpy_prefill_memory_reward', agent.memory_reward)
#     np.save('numpy_prefill_memory_action', agent.memory_action)
#     np.save('numpy_prefill_memory_done', agent.memory_done)
#     np.save('numpy_prefill_memory_next', agent.memory_next)
#
# #####
#
# big_counter = 0
# rewards = []
# loss_list = []
#
# for episode in range(10000):
#
#     #reset env, get first four images
#     state = env.reset()
#
#     #retrieve first four images
#     counter = 0
#     no_reward_counter = 0
#     tmp_images = []
#
#     for i in range(0, 5):
#         action = env.action_space.sample()
#         state, reward, done, _ = env.step(action)
#         img = image_processor.preprocess(state, dims)
#         current_state = np.roll(current_state, 1, axis=0)
#         current_state[0] = img
#
#     done = False
#     tot_reward = 0
#     start = time.time()
#     state = env.reset()
#     print("start real")
#     #lives = env.unwrapped.ale.lives()
#     while not done and no_reward_counter < 400:
#
#         previous_state = current_state
#         e = random.random()
#         if e < agent.epsilon:
#             action = env.action_space.sample()
#         else:
#             action = agent.findAction(np.expand_dims(current_state, axis=0))
#
#         state, reward, done, _ = env.step(action)
#         img = image_processor.preprocess(state, dims)
#         counter += 1
#         big_counter += 1
#         if reward == 0:
#             no_reward_counter += 1
#         else:
#             no_reward_counter = 0
#
#         current_state = np.roll(current_state, 1, axis=0)
#         current_state[0] = img
#
#         #lost_life_or_done = env.unwrapped.ale.lives() < lives or done
#
#         agent.addToMemory(previous_state, action, reward, current_state, done, big_counter)
#         tot_reward += reward
#         loss = agent.fitBatch(reward, done, 32)
#         loss_list.append(loss)
#         if (big_counter + 1) % 4000 == 0:
#             print("train target", big_counter)
#             agent.target_train()
#
#     rewards.append(tot_reward)
#         #env.render()
#     agent.epsilon = 1.0 - ((1.0 - agent.epsilon_min) * (episode/8000 if episode < 8000 else 1))
#     #agent.target_train()
#     end = time.time()
#     print()
#     print("decision vector", agent.getPredictionVector())
#     print("frames played", counter)
#     print("total frames played", big_counter)
#     print("finished cleanly", done if done else no_reward_counter)
#     print("epsilon: ", agent.epsilon)
#     print("real end time: ", end - start)
#     print("average per step", (end - start)/counter)
#     print("image process time per step", image_processor.processTime)
#     print("addToMemory time per step", agent.addToMemoryTime)
#     print("fitBatch time per step", agent.fitBatchTime)
#     print("findAction time per step", agent.findActionTime)
#     agent.saveToDisk(stored_model, episode)
#
#     if len(rewards) % 10 == 0:
#         # Scalar summaries for tensorboard
#         #if frame_number > REPLAY_MEMORY_START_SIZE:
#         summ = agent.sess.run(PERFORMANCE_SUMMARIES,
#                         feed_dict={LOSS_PH: np.mean(loss_list),
#                                    REWARD_PH: np.mean(rewards[-100:])})
#
#         SUMM_WRITER.add_summary(summ, big_counter)
#         loss_list = []
#         # Histogramm summaries for tensorboard
#         summ_param = agent.sess.run(PARAM_SUMMARIES)
#         SUMM_WRITER.add_summary(summ_param, big_counter)
#
#         print(len(rewards), big_counter, np.mean(rewards[-100:]))
#         # with open('rewards.dat', 'a') as reward_file:
#         #     print(len(rewards), big_counter,
#         #           np.mean(rewards[-100:]), file=reward_file)
#
#
#     #Print score
#     print("episode: {}/10000, score: {}"
#             .format(episode, tot_reward))
#
# agent.saveToDisk(stored_model);
#
# env.close()

# OpenGym CartPole-v0 with A3C on GPU
# -----------------------------------
#
# A3C implementation with GPU optimizer threads.
#
# Made as part of blog series Let's make an A3C, available at
# https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/
#
# author: Jaromir Janisch, 2017

import numpy as np
import tensorflow as tf

import gym, time, random, threading

from keras.models import *
from keras.layers import *
from keras import backend as K
from brain import Brain

# -- constants
ENV = 'CartPole-v0'

RUN_TIME = 30
THREADS = 8
OPTIMIZERS = 2
THREAD_DELAY = 0.001

GAMMA = 0.99

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.4
EPS_STOP = .15
EPS_STEPS = 75000

MIN_BATCH = 32
LEARNING_RATE = 5e-3

LOSS_V = .5  # v loss coefficient
LOSS_ENTROPY = .01  # entropy coefficient


# ---------
frames = 0


class Agent:
    def __init__(self, eps_start, eps_end, eps_steps):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps

        self.memory = []  # used for n_step return
        self.R = 0.

    def getEpsilon(self):
        if (frames >= self.eps_steps):
            return self.eps_end
        else:
            return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps  # linearly interpolate

    def act(self, s):
        eps = self.getEpsilon()
        global frames;
        frames = frames + 1

        if random.random() < eps:
            return random.randint(0, NUM_ACTIONS - 1)

        else:
            s = np.array([s])
            p = brain.predict_p(s)[0]

            # a = np.argmax(p)
            a = np.random.choice(NUM_ACTIONS, p=p)

            return a

    def train(self, s, a, r, s_):
        def get_sample(memory, n):
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n - 1]

            return s, a, self.R, s_

        a_cats = np.zeros(NUM_ACTIONS)  # turn action into one-hot representation
        a_cats[a] = 1

        self.memory.append((s, a_cats, r, s_))

        self.R = (self.R + r * GAMMA_N) / GAMMA

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                brain.train_push(s, a, r, s_)

                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)

            self.R = 0

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            brain.train_push(s, a, r, s_)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)

            # possible edge case - if an episode ends in <N steps, the computation is incorrect

# ---------
class Optimizer(threading.Thread):
    stop_signal = False

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while not self.stop_signal:
            brain.optimize()

    def stop(self):
        self.stop_signal = True


# -- main
env_test = Environment(render=True, eps_start=0., eps_end=0.)
NUM_STATE = env_test.env.observation_space.shape[0]
NUM_ACTIONS = env_test.env.action_space.n
NONE_STATE = np.zeros(NUM_STATE)
dims = (84,84)

brain = Brain(state_size=(4,84,84), action_size=NUM_ACTIONS, loss_entropy=LOSS_ENTROPY,
              loss_v=LOSS_ENTROPY, learning_rate=LEARNING_RATE, min_batch=MIN_BATCH,
              gamma_n=GAMMA_N)  # brain is global in A3C

envs = [Environment() for i in range(THREADS)]
opts = [Optimizer() for i in range(OPTIMIZERS)]

for o in opts:
    o.start()

for e in envs:
    e.start()

time.sleep(RUN_TIME)

for e in envs:
    e.stop()
for e in envs:
    e.join()

for o in opts:
    o.stop()
for o in opts:
    o.join()

print("Training finished")
env_test.run()