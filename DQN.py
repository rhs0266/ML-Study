import tensorflow as tf
import numpy as np

class DQN(object):
    def __init__(self, feature_depth, units_list):
        self.feature_depth = feature_depth
        self.model = tf.keras.Sequential(name='DQN')
        self.gamma = 0.99
        for i, units in enumerate(units_list):
            self.model.add(tf.keras.layers.Dense(units=units, activation=tf.nn.leaky_relu, name=f"generator_dense_{i}"))
        self.model.add(tf.keras.layers.Dense(units=feature_depth, activation=tf.nn.tanh, name=f"generator_dense_final"))
        
    def forward(self, state):
        return self.model(state, training=False)

    def loss(self, pkl):
        q = tf.math.reduce_max(self.model(np.array([p.state for p in pkl]), training=True), axis=1)
        q_exp = np.array([p.reward for p in pkl]) + self.gamma * tf.math.reduce_max(self.model(np.array([p.next_state for p in pkl]), training=False), axis=1)
        print(q)
        print(q_exp)

        return tf.keras.losses.huber(q_exp, q)