import gym 
import numpy as np 
from collections import defaultdict


env = gym.make("Blackjack-v1")


def policy(state):
	return 0 if state[0] > 20 else 1 


state = env.reset()


def generate_episode(policy):

	num_iterations  = 100 
	state  = env.reset()

	episode = []


	for i in range(num_iterations):
		print("Running ",i)

		action = policy(state)

		next_state, reward, done , info = env.step(action)


		episode.append((state,action,reward))

		if done:
			break 

		state =  next_state

	return episode




total_return = defaultdict(float)
N = defaultdict(int)


iterations = 50_000


for x in range(iterations):


	episode = generate_episode(policy)

	states,actions, rewards =  zip(*episode)


	for t,state in enumerate(states):

		R = (sum(rewards[t:]))

		total_return[state] += R 

		N[state] +=1 





