import gym
print('imported ')

if __name__ == '__main__':
	env = gym.make('CartPole-v0')
	state = env.reset()
	env.render()
	steps = 0
	total_reward = 0
	print('part 1')

	while True:
		env.render()
		state,reward,done,info = env.step(env.action_space.sample())
		steps += 1
		total_reward += reward
		if done:
			break

	print(f'Done at {steps} and Reward is {total_reward}')