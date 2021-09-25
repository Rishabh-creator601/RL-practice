import numpy as np
import gym 


env = gym.make('FrozenLake-v1')
env.reset()

env.render()


def value_iteration(env,gamma=1.0):
	value_table = np.zeros(env.observation_space.n) # 16
	no_of_iterations=100
	threshhold=1e-2


	for i in range(no_of_iterations):
		copy_table = np.copy(value_table)

		for state in range(env.observation_space.n):
			Q_value = []

			for action in range(env.action_space.n): #4
				next_states_reward = []

				for next_str in env.P[state][action]:
					trans_prob,next_state,reward_prob ,_ = next_str
					next_states_reward.append((trans_prob * (reward_prob + gamma * copy_table[next_state])))
				Q_value.append(np.sum(next_states_reward))
			value_table[state] = np.max(Q_value)

		if(np.sum(np.fabs(copy_table - value_table <= threshhold))):
			print(f'value iteration converged at {i+1}')
			break
	return value_table



def extract_policy(value_table,gamma=1.0):
	policy = np.zeros(env.observation_space.n)

	for state in range(env.observation_space.n):
		Q_table = np.zeros(env.action_space.n)

		for action in range(env.action_space.n):
			for next_str in env.P[state][action]:
				trans_prob, next_state, reward_prob, _ = next_str
				Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))

		policy[state] = np.argmax(Q_table)
	return policy 



optimal_func = value_iteration(env)
optimal_policy = extract_policy(optimal_func)
print(optimal_policy)
				 







	