import gym 
import numpy as np


print("Imported...")

env = gym.make("FrozenLake-v1")

def optimal_value_iteration(env):

	gamma = 1.0
	value_table =  np.zeros(env.observation_space.n)
	threshold =1e-20

	num_itertions =1000


	


	for i in range(num_itertions):
		updated_table = np.copy( value_table)

		for s in range(env.observation_space.n):

			Q_values = [sum([prob*(r + gamma * updated_table[s_])
				              for prob,s_,r,_ in env.P[s][a]])
			                      for a in range(env.action_space.n)]



			value_table[s] = max(Q_values)
		if (np.sum(np.fabs(updated_table - value_table)) <= threshold):
			break 
	return value_table







def extract_policy(value_table):

	gamma = 1.0
	policy  = np.zeros(env.observation_space.n)



	for s in range(env.observation_space.n):

		Q_values = [sum([prob*(r+gamma*value_table[s_])
			            for prob,s_,r,_ in env.P[s][a]])
		             for a in range(env.action_space.n)]


		policy[s] = np.argmax(np.array(Q_values))

	return policy 





table = optimal_value_iteration(env)
path = extract_policy(table)


print(path)
print(env.render())


print(env.observation_space.n)
