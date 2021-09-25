import gym 
import random


env = gym.make('Taxi-v3')



q = {}
for s in range(env.observation_space.n):
	for a in range(env.action_space.n):
		q[(s,a)] = 0.0




def update_q_table(prev_state,action,reward,next_state,alpha,gamma):
	qa = max([q[(next_state, a)] for a in range(env.action_space.n)])
	q[(prev_state,action)] += alpha * (reward + gamma * qa - q[(prev_state,action)])



def epilson_greedy_policy(state,epilson):
	if random.uniform(0,1) < epilson:
		return env.action_space.sample()
	else:
		return max(list(range(env.action_space.n)),key = lambda x : q[(state,x)])


alpha = 0.4
gamma = 1.0
epilson = 0.017

iteratios = 700

for i in range(iteratios):
	print(f'{i} done ')
	r = 0
	prev_state = env.reset()
	while True:
		env.render()
		action = epilson_greedy_policy(prev_state,epilson)
		next_state,reward,done,info = env.step(action)
		update_q_table(prev_state,action,reward,next_state,alpha,gamma)

		prev_state = next_state
		r += reward

		if done:
			break
	print(f'TOTAL REWARD IS {r}')
env.close()


