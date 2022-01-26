import numpy as np 
from collections import defaultdict
import gym 


env = gym.make("FrozenLake-v1")


state_size = env.observation_space.n
action_size = env.action_space.n 


Q = {}


for s in range(state_size):
	for a in range(action_size):
		Q[(s,a)] =  0.0


def epsilon_greedy(state,epsilon=0.8):
	if np.random.uniform(0,1) <= epsilon:
		return env.action_space.sample()

	else:
		return max(list(range(env.action_space.n)) , key = lambda x: Q[(state,x)])






alpha = 0.95
gamma = 0.90
epsilon = 0.8 


for i in range(20_000):


	state = env.reset()
	total = []

	a = epsilon_greedy(state,epsilon)

	for j in range(1000):


		next_state,reward,done, _ =  env.step(a)
		a_ =  epsilon_greedy(next_state,epsilon)

		Q[(s,a)] += alpha * (reward + gamma * (Q[(next_state,a_)] -  Q[(state,a)]))
		total.append(reward)

		state = next_state

		a =  a_ 

		if done:
			print(f"Episode : {i} Reward : {reward}")
			break 






