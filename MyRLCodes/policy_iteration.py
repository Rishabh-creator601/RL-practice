import gym 
import numpy as np 



env = gym.make("FrozenLake-v1")

def optimal_policy(policy):
	num_iterations =1000
	threshold = 1e-20
	value_table = np.zeros(env.observation_space.n)
	gamma = 1.0


	for i in range(num_iterations):
		updated_table = np.copy(value_table)

		for s in range(env.observation_space.n):
			a = policy[s]

			value_table[s] = sum([prob *(r + gamma * updated_table[s_] )
				           for prob, s_, r, _ in env.P[s][a]])
			             





		if (np.sum((np.fabs(updated_table - value_table))) <= threshold):
			break 


	return value_table





def extract_policy(value_table):
	num_iterations =1000
	threshold = 1e-20
	gamma = 1.0
	policy = np.zeros(env.observation_space.n)




	for s in range(env.observation_space.n):

		Q_values = [sum([prob *(r + gamma * value_table[s_] )
				           for prob, s_, r, _ in env.P[s][a]])
			             for a in range(env.action_space.n)]


		policy[s] =np.argmax(np.array(Q_values))


	return policy 













def policy_iteration(env):

	num_iter = 1000
	policy = np.zeros(env.observation_space.n)

	for x in range(num_iter):

		value_function = optimal_policy(policy)
		new_policy = extract_policy(value_function)


		if (np.all(policy == new_policy)):
			break 

		policy = new_policy

	return policy 




print(policy_iteration(env))