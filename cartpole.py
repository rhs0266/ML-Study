import random

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from ReplayMemory import ReplayMemory
from DQN import DQN

env = gym.make('CartPole-v0')

# init
plt.ion()
dqn = DQN(2, [4, 256])

def sample_action(state, epsilon):
    global dqn
    if random.random() < epsilon:
        return random.randint(0, 1)
    return tf.math.argmax(dqn.forward(state), axis=1).numpy()[0]


def optimizing(trns):
    global dqn_opt
    with tf.GradientTape() as tape:
        # states, actions, next_states, rewards
        loss = dqn.loss(trns)
        grad = tape.gradient(target=loss, sources=dqn.model.trainable_variables)
        dqn_opt.apply_gradients(zip(grad, dqn.model.trainable_variables))


def training():
    rep_mem = ReplayMemory(1024)
    
    term = 100
    epsilon = 0.1
    transition_count = 0
    TS = 512
    for episode in range(1, term * 10000):
        # TODO: Supply multi-environment for faster learning
        obs = env.reset()
        for _ in range(1, 100000):
            # run episode
            env.render()

            # sample action follows eps-greedy method
            action = sample_action(obs.reshape([-1] + list(obs.shape)), epsilon)

            # stepping with selected action
            nxt, reward, done, _ = env.step(action)

            # push transition information into the replay memory
            if done:
                reward = 0
            rep_mem.push(obs, action, nxt, reward)
            obs = nxt

            transition_count += 1
            if transition_count % TS == 0:
                # optimizing
                optimizing(rep_mem.sample(TS))

            if done:
                break
        
        if episode % term == 0:
            # create checkpoint
            pass
    env.close()
def evaluate():
    pass

if __name__ == "__main__":
    dqn_opt = tf.keras.optimizers.Adam(learning_rate=2e-4)
    training()
    