import numpy as np
import tensorflow as tf

import gym, time, random, threading

from keras.models import *
from keras.layers import *
from keras.initializers import Orthogonal
from keras import backend as K
import os

# ---------
class Brain:
    train_queue = [[], [], [], []]  # s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()
    lock_file = threading.Lock()

    def __init__(self, state_size, action_size, loss_entropy, loss_v, learning_rate, min_batch, gamma):
        self.GAMMA = gamma
        self.NONE_STATE = np.zeros(state_size)
        self.MIN_BATCH = min_batch
        self.LEARNING_RATE = learning_rate
        self.LOSS_V = loss_v
        self.LOSS_ENTROPY = loss_entropy
        self.state_size = state_size
        self.action_size = action_size
        self.decay_steps = 0
        self.decay_max = 10000
        self.session = tf.Session()
        self.rewards = []
        self.total_episodes = 0
        self.max_reward = 0

        self.session = tf.Session()
        K.set_session(self.session)

        self.make_model()

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()

        self.default_graph.finalize()  # avoid modifications
        self.writer = tf.summary.FileWriter(".log/run_9",
                                            self.session.graph)

    # Copied from https://github.com/MG2033/A2C/blob/master/layers.py
    def openai_entropy(self, logits):
        # Entropy proposed by OpenAI in their A2C baseline
        a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)

    def make_model(self):

        self.input_tensor = tf.placeholder(tf.float32, shape=(None, 4, 84, 84))

        self.R_tensor = tf.placeholder(tf.float32, shape=[None])
        self.A_tensor = tf.placeholder(tf.int32, shape=[None])
        # We have advantage_tensor be separate in order so the gradients for
        # the actor don't go through the critic
        self.advantage_tensor = tf.placeholder(tf.float32, shape=[None])

        self.global_step = tf.Variable(0, trainable=False)

        self.lr_tensor = tf.placeholder(tf.float32, shape=[])

        with tf.variable_scope("policy"):
            conv1 = tf.layers.conv2d(
                    inputs=(tf.cast(self.input_tensor, tf.float32) / 255),
                    filters=32,
                    kernel_size=[8,8],
                    data_format="channels_first",
                    strides=4,
                    padding='valid',    # 'same'?
                    activation=tf.nn.relu,
                    kernel_initializer=tf.initializers.orthogonal()
            )
            conv2 = tf.layers.conv2d(
                    inputs=conv1,
                    filters=64,
                    kernel_size=[4,4],
                    data_format="channels_first",
                    strides=2,
                    padding='valid',    # 'same'?
                    activation=tf.nn.relu,
                    kernel_initializer=tf.initializers.orthogonal()
            )
            conv3 = tf.layers.conv2d(
                    inputs=conv2,
                    filters=64,
                    kernel_size=[3,3],
                    data_format="channels_first",
                    strides=1,
                    padding='valid',    # 'same'?
                    activation=tf.nn.relu,
                    kernel_initializer=tf.initializers.orthogonal()
            )

            conv3_flat = tf.contrib.layers.flatten(conv3)

            fully_connected = tf.contrib.layers.fully_connected(
                    conv3_flat,
                    512,
                    activation_fn=tf.nn.relu,
                    weights_initializer=tf.initializers.orthogonal()

            )

            self.actor_policy_logits = tf.contrib.layers.fully_connected(
                    fully_connected,
                    self.action_size,
                    activation_fn=None,
                    weights_initializer=tf.initializers.orthogonal()
            )

            self.actor_output_tensor = tf.nn.softmax(self.actor_policy_logits)


            self.critic_output_tensor = tf.contrib.layers.fully_connected(
                    fully_connected,
                    1,
                    activation_fn=None,
                    weights_initializer=tf.initializers.orthogonal()
            )

        neg_log_action_probabilities = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.actor_policy_logits,
            labels=self.A_tensor
        )

        # sparse_softmax_cross_entropy is already negative, so don't need - here
        self.actor_loss = tf.reduce_mean(self.advantage_tensor * neg_log_action_probabilities)
        self.critic_loss = tf.reduce_mean(tf.square(self.R_tensor - tf.squeeze(self.critic_output_tensor)) / 2)
        self.temp_entropy = tf.reduce_mean(self.openai_entropy(self.actor_policy_logits))
        # self.entropy = self.temp_entropy
        self.entropy = tf.cond(tf.less(self.temp_entropy, 0.8), lambda: 0.8, lambda: self.temp_entropy)
        # self.entropy = self.LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)
        self.loss = self.actor_loss + 0.5 * self.critic_loss - 0.01 * self.entropy

        print(self.loss)

        with tf.variable_scope("policy"):
            params = tf.trainable_variables()
        grads = tf.gradients(self.loss, params)
        grads, grad_norm = tf.clip_by_global_norm(grads, 0.5)
        grads = list(zip(grads, params))

        self.optimizer = tf.train.RMSPropOptimizer(self.lr_tensor,
            decay=0.99, epsilon=1e-5)

        self.train_both = self.optimizer.apply_gradients(grads)

        self.saver = tf.train.Saver(max_to_keep=1)

        self.session.run(tf.global_variables_initializer())

        if os.path.exists('saves'):
            self.saver.restore(self.session, 'saves/')

    # def _build_model(self):
    #     print((None,) + self.state_size)
    #     l_input = Input(batch_shape=(None,) + self.state_size)
    #     #normalize = BatchNormalization()(l_input)
    #     conv_1 = Conv2D(filters=32,
    #                      kernel_size=8,
    #                      strides=(4, 4),
    #                      padding="valid",
    #                      activation="relu",
    #                      kernel_initializer=Orthogonal(),
    #                      use_bias=False,
    #                      data_format="channels_first")(l_input)
    #     conv_2 = Conv2D(filters=64,
    #                     kernel_size=4,
    #                      strides=(2, 2),
    #                      padding="valid",
    #                      activation="relu",
    #                      kernel_initializer=Orthogonal(),
    #                      use_bias=False,
    #                      data_format="channels_first")(conv_1)
    #     conv_3 = Conv2D(filters=64,
    #                     kernel_size=1,
    #                     strides=(1, 1),
    #                     padding="valid",
    #                     activation="relu",
    #                     kernel_initializer=Orthogonal(),
    #                     use_bias=False,
    #                     data_format="channels_first")(conv_2)
    #     flatten = Flatten()(conv_3)
    #     dense_1 = Dense(512, activation='relu')(flatten)
    #
    #     out_actions = Dense(self.action_size, activation='softmax')(dense_1)
    #     out_value = Dense(1, activation='linear')(dense_1)
    #
    #     model = Model(inputs=[l_input], outputs=[out_actions, out_value])
    #     model._make_predict_function()  # have to initialize before threading
    #
    #     return model
    #
    # def _build_graph(self, model):
    #     s_t = tf.placeholder(tf.float32, shape=(None,) + self.state_size)
    #     a_t = tf.placeholder(tf.float32, shape=(None, self.action_size))
    #     r_t = tf.placeholder(tf.float32, shape=(None, 1))  # not immediate, but discounted n step reward
    #
    #     p, v = model(s_t)
    #
    #     log_prob = tf.log(tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
    #     advantage = r_t - v
    #
    #     loss_policy = - log_prob * tf.stop_gradient(advantage)  # maximize policy
    #     loss_value = self.LOSS_V * tf.square(advantage)  # minimize value error
    #     entropy = self.LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1,
    #                                                 keep_dims=True)  # maximize entropy (regularization)
    #
    #     loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)
    #
    #     optimizer = tf.train.RMSPropOptimizer(self.LEARNING_RATE, decay=.99)
    #     minimize = optimizer.minimize(loss_total)
    #
    #     return s_t, a_t, r_t, minimize

    def decay_lr(self):
        decayed = self.LEARNING_RATE * ((1 - self.total_episodes / self.decay_max) if self.total_episodes < self.decay_max else 1)
        return decayed

    def optimize(self):
        if len(self.train_queue[0]) < self.MIN_BATCH:
            time.sleep(0)  # yield
            return

        with self.lock_queue:
            if len(self.train_queue[0]) < self.MIN_BATCH:  # more thread could have passed without lock
                return  # we can't yield inside lock

            s, a, r, done = self.train_queue
            self.train_queue = [[], [], [], []]

        for i in range(len(s)):
            last_state = s[i][-1]

            last_value = self.session.run(
                self.critic_output_tensor,
                feed_dict={
                    self.input_tensor: np.array([last_state])
                },
            )

            values = self.session.run(
                self.critic_output_tensor,
                feed_dict={
                    self.input_tensor: s[i]
                },
            )
            values_array = np.array(values)

            R = np.zeros_like(values_array)

            # if len(s) > 5 * self.MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))

            # Note!!! Only the last state value is wrapped into the rollout! Not
            # the value for each state at each time-step!
            if done[i][-1] == 1:
                cumulative_discounted = 0
            else:
                cumulative_discounted = last_value

            for t in range(len(r[i]) - 1):
                cumulative_discounted = r[i][t] + self.GAMMA * cumulative_discounted
                R[t] = cumulative_discounted

            decayed_lr = self.decay_lr()

            advantage = np.ndarray.flatten(R - values)

            actor_loss, critic_loss, entropy, _ = self.session.run(
                [self.actor_loss, self.critic_loss, self.entropy, self.train_both],
                feed_dict={
                    self.lr_tensor: decayed_lr,
                    self.input_tensor: s[i],
                    self.A_tensor: a[i],
                    self.R_tensor: R.reshape((-1)),
                    self.advantage_tensor: advantage,
                },
            )

            if self.total_episodes > 0:

                reward_summary = tf.Summary(value=[tf.Summary.Value(tag="average_reward",
                                                                    simple_value=np.mean(self.rewards[-100:]))])
                self.writer.add_summary(reward_summary, self.total_episodes)

                reward_summary = tf.Summary(value=[tf.Summary.Value(tag="actor_loss",
                                                                    simple_value=actor_loss)])
                self.writer.add_summary(reward_summary, self.total_episodes)

                reward_summary = tf.Summary(value=[tf.Summary.Value(tag="max_reward",
                                                                    simple_value=self.max_reward)])
                self.writer.add_summary(reward_summary, self.total_episodes)

                reward_summary = tf.Summary(value=[tf.Summary.Value(tag="critic_loss",
                                                                    simple_value=critic_loss)])
                self.writer.add_summary(reward_summary, self.total_episodes)

                reward_summary = tf.Summary(value=[tf.Summary.Value(tag="entropy",
                                                                    simple_value=entropy)])
                self.writer.add_summary(reward_summary, self.total_episodes)
        if self.total_episodes % 50 == 0:
            print("save at ", self.total_episodes)
            with self.lock_file:
                if not os.path.exists('saves'):
                    os.mkdir('saves')
                self.saver.save(self.session, 'saves/')
            print("saved. Length queue: ", len(self.train_queue[0]) )

        # v = self.predict_v(s_)
        # r = r + self.GAMMA_N * v * s_mask  # set v to 0 where s_ is terminal state
        #
        # s_t, a_t, r_t, minimize = self.graph
        # self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})

    def train_push(self, s, a, r, done):
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)
            self.train_queue[3].append(done)
            if len(self.train_queue[0]) > 2000:
                print(len(self.train_queue))

    # def predict(self, s):
    #     with self.default_graph.as_default():
    #         p, v = self.model.predict(s)
    #         return p, v

    def predict_p(self, s):
        with self.default_graph.as_default():
            probs = self.session.run(
                self.actor_output_tensor,
                feed_dict={
                    self.input_tensor: np.array(s)
                },
            )
            return probs

    # def predict_v(self, s):
    #     with self.default_graph.as_default():
    #         p, v = self.model.predict(s)
    #         return v

    def add_rewards(self, reward):
        with self.lock_file:
            self.rewards.append(reward)
            self.max_reward = reward if reward > self.max_reward else self.max_reward
            self.total_episodes += 1
