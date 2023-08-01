import random
import numpy as np
import tensorflow as tf
from collections import deque
from threading import Thread
from keras.models import Sequential
from keras.layers import Dense, Conv2D, AveragePooling2D, Activation, Flatten
import time

class DQLAgent:
    """Class representing the RL agent."""

    def __init__(self):
        self.Discount = 0.99
        self.sample_batch_size = 16
        self.update_target = 5
        self.prediction_batch_size = 1
        self.training_batch_size = self.sample_batch_size // 4
        self.replay_memory_size = 100
        self.min_replay_memory_size = 50
        self.replay_memory = deque(maxlen=self.replay_memory_size)
        self.im_height = 71
        self.im_width = 71
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.graph = tf.get_default_graph()
        self.end_state = False
        self.last_logged_episode = 0
        self.initialise_training = False
        self.target_update_counter = 0

    def create_model(self) -> Sequential:
        """Create the neural network model for model training.

        Returns:
            Sequential: Keras sequential model.
        """
        model = Sequential()
        model.add(Conv2D(128, (3, 3), input_shape=(self.im_height, self.im_width, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Flatten())
        model.add(Dense(1024, activation="relu"))
        model.add(Dense(512, activation="relu"))
        model.add(Dense(7, activation='linear'))
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        return model

    def update_replay_memory(self, transistion: tuple) -> None:
        """Update the replay memory buffer.

        Args:
            transistion (tuple): Tuple of (current_state, action, reward, new_state, done).
        """
        self.replay_memory.append(transistion)

    def train(self) -> None:
        """Train the model."""
        x = []
        y = []
        if len(self.replay_memory) < self.min_replay_memory_size:
            return
        sample_batch = random.sample(self.replay_memory, self.sample_batch_size)
        for transition in sample_batch:
            current_states = np.array([transition[0]]) / 255.0
            with self.graph.as_default():
                current_qlist = self.model.predict(current_states, self.prediction_batch_size)
            future_states = np.array([transition[3]]) / 255.0
            with self.graph.as_default():
                future_qlist = self.target_model.predict(future_states, self.prediction_batch_size)
            for index, (current_state, action, reward, new_state, done) in enumerate(sample_batch):
                if not done:
                    q_max = np.max(future_qlist[index])
                    q_new = reward + self.Discount * q_max
                else:
                    q_new = reward

                target_qlists = current_qlist[index]
                target_qlists[action] = q_new
                x.append(current_state)
                y.append(target_qlists)
        with self.graph.as_default():
            self.model.fit(np.array(x) / 255.0, np.array(y), batch_size=self.training_batch_size, verbose=0, shuffle=False)
        if self.target_update_counter > self.update_target:
            self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter += 1

    def get_qs(self, state: np.ndarray) -> np.ndarray:
        """Get the Q-values.

        Args:
            state (np.ndarray): Input state.

        Returns:
            np.ndarray: Q-values.
        """
        state = (np.array(state).reshape(-1, *state.shape) / 255)[0]
        return self.model.predict(state)

    def train_in_loop(self) -> None:
        """Train the model in a continuous loop."""
        with self.graph.as_default():
            self.model.fit(x, y, verbose=False, batch_size=1)
        self.initialise_training = True
        while True:
            if self.end_state:
                return
            self.train()
            time.sleep(0.01)
