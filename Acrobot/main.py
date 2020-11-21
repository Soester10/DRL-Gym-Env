import math
import gym
from Acrobot_DQN import Acrobot_DQN
from Acrobot_Game import Acrobot_Game


env = gym.make("Acrobot-v1")

min_len4train = 100
max_len4train = 50_000
DISCOUNT = 0.90
min_batch = 64
Batch_Size = 32
SHOW_EVERY = 200
UPDATE_SECONDARY_WEIGHTS = False
UPDATE_SECONDARY_WEIGHTS_NUM = 4

EPISODES = 1000

epsilon = 1
epsilon_mul_value = math.log(0.01, 10)/(EPISODES * 0.8)
epsilon_mul_value = math.pow(10, epsilon_mul_value)



#main

Neuron = Acrobot_DQN(max_len4train, UPDATE_SECONDARY_WEIGHTS, min_batch, min_len4train, DISCOUNT, Batch_Size)

Acrobot_Game(Neuron, env, EPISODES, min_len4train, epsilon, epsilon_mul_value, SHOW_EVERY, UPDATE_SECONDARY_WEIGHTS_NUM)

#saving model and weights
Neuron.model.save_weights("weights")
Neuron.model.save("model.h5")
