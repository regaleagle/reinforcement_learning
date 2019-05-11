import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
from keras.initializers import RandomUniform
from keras.callbacks import TensorBoard
from keras import backend as K
from tensorboard_remake import TensorBoardCustom
import tensorflow as tf

class Agent:

    def __init__(self, state_size, action_size, model, target_model, queue_length, stored_model="nothing", run_name = "run_1"):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95
        self.epsilon = 1.0
        self.queue_length = queue_length
        self.memory_current = np.zeros((queue_length, state_size[0], state_size[1], state_size[2]), dtype=np.uint8)
        self.memory_next = np.zeros((queue_length, state_size[0], state_size[1], state_size[2]), dtype=np.uint8)
        self.memory_action = np.zeros(queue_length, dtype=np.uint8)
        self.memory_reward = np.zeros(queue_length, dtype=np.uint8)
        self.memory_done = np.zeros(queue_length, dtype=bool)
        self.epsilon_min = 0.3
        # self.epsilon_decay = 0.005
        self.model = model.model # self._buildModel() if stored_model == "nothing" else load_model(stored_model)
        self.target_model = target_model.model # self._buildModel() if stored_model == "nothing" else load_model(stored_model)
        self.tau = .05
        self.tensorboard = TensorBoardCustom('.log/' + run_name + '/')
        self.session = tf.Session()
        K.set_session(self.session)

        self.addToMemoryTime = 0
        self.fitBatchTime = 0
        self.findActionTime = 0

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()

        self.default_graph.finalize()    # avoid modifications

    #
    # def _buildModel(self):
    #     model = Sequential()
    #     model.add(Conv2D(32,
    #                             8,
    #                             strides=(4, 4),
    #                             padding="valid",
    #                             activation="relu",
    #                             input_shape=self.state_size,
    #                             data_format="channels_first"))
    #     model.add(Conv2D(64,
    #                             4,
    #                             strides=(2, 2),
    #                             padding="valid",
    #                             activation="relu",
    #                             input_shape=self.state_size,
    #                             data_format="channels_first"))
    #     model.add(Conv2D(64,
    #                      4,
    #                      strides=(1, 1),
    #                      padding="valid",
    #                      activation="relu",
    #                      input_shape=self.state_size,
    #                      data_format="channels_first"))
    #     model.add(Flatten())
    #     model.add(Dense(512, activation="relu"))
    #     model.add(Dense(self.action_size, activation="linear", kernel_initializer=RandomUniform(minval=0, maxval=0.0001), bias_initializer=RandomUniform(minval=0, maxval=0.0001)))
    #     model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse', metrics=['accuracy'])
    #     return model

    def fitBatch(self, reward, done, batch_size=32):

        if len(self.memory_current) < batch_size:
            return

        #start = time.time()

        indexes = np.random.randint(low=0, high=self.queue_length, size=batch_size, dtype=int)
        targets = np.zeros((batch_size,self.action_size))
        future_q_array = self.target_model.predict(self.memory_next[indexes])

        count = 0
        for i in indexes:
            target = np.zeros(4)
            if self.memory_done[i]:
                target[self.memory_action[i]] = self.memory_reward[i]
            else:
                Q_future = max(future_q_array[count])
                target[self.memory_action[i]] = self.memory_reward[i] + Q_future * self.gamma
            targets[count] = target
            count += 1
        self.tensorboard.episode_score += reward
        self.tensorboard.done = done
        history = self.model.fit(self.memory_current[indexes], targets, epochs=1, batch_size=batch_size, verbose=0, callbacks=[self.tensorboard])
        return history.history['loss']
        # end = time.time()
        # self.fitBatchTime = (self.fitBatchTime + (end - start)) / 2

    def findAction(self, state):
        #start = time.time()
        return np.argmax(self.model.predict(state))
        #end = time.time()
        #self.findActionTime = (self.findActionTime + (end - start)) / 2
        #return thing

    def getPredictionVector(self):
        sample_index = np.random.randint(low=0, high=self.queue_length, dtype=int)
        #state, action, reward, new_state, done = samples[0]
        guessVector = self.model.predict(np.expand_dims(self.memory_current[sample_index], axis=0))[0]
        return guessVector

    def addToMemory(self, state, action, reward, next_state, done, counter):
        #start = time.time()
        index = counter % self.queue_length
        self.memory_current[index] = state
        self.memory_next[index] = next_state
        self.memory_action[index] = action
        self.memory_reward[index] = reward
        self.memory_done[index] = done
        #end = time.time()
        #self.addToMemoryTime = (self.addToMemoryTime + (end - start)) / 2


    def saveToDisk(self, filename, episode, epsilon):
        if (episode +1) % 10 == 0:
            self.target_model.save(filename + ".h5")
            with open(filename +".csv", "w") as text_file:
                print(f"{episode}, {epsilon}", file=text_file)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)
