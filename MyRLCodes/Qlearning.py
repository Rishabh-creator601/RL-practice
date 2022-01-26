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






alpha = 0.85
gamma = 0.90
epsilon = 0.8 


for i in range(20_000):


	state = env.reset()

	for j in range(1000):

		a =  epsilon_greedy(state,epsilon)


		next_state,reward,done, _ =  env.step(a)
		a_ = np.argmax(Q[next_state,a] for a in range(env.action_space.n) )

		Q[(s,a)] += alpha * (reward + gamma * (Q[(next_state,a_)] -  Q[(state,a)]))

		state = next_state

		if done:
			print(f"Episode : {i} Reward : {reward}")
			break 






