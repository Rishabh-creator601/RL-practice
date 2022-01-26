import gym 
import numpy as np 


env = gym.make("Taxi-v3")
q_table = np.zeros([env.observation_space.n,env.action_space.n])



alpha = 0.1
gamma  = 0.6


def get_action(state,epsilon):
	if np.random.normal(0,1) > epsilon:
		return env.action_space.sample()

	else:
		return np.argmax(q_table[state])






for i in range(5000):

	state = env.reset()

	penalities  = 0.0
	total  = 0.0



	done = False 


	while not done:

		action = get_action(state,0.8)

		next_state,reward,done,info = env.step(action)

		total += reward
		if reward == -10:
			penalities += 1

		q_table[state][action]  =  ( 1- alpha)*q_table[state][action] + alpha *(reward + gamma * np.max(q_table[next_state]))


		state = next_state

	
	print(f"Epoch : {i + 1}")
	print(f"Reward : {reward} Total : {total} Penalties:{penalities} ")



print("Training Finished !")



