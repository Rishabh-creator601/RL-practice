import gym 
import random


class RandomActionWrapper(gym.ActionWrapper):
	def __init__(self,env,epsilon=0.1):
		super(RandomActionWrapper,self).__init__(env)
		self.epsilon =epsilon

	def action(self,action):
		if random.random() < self.epsilon:
			print('random')
			return self.env.action_space.sample()
		return action

if __name__ == '__main__':
	env = RandomActionWrapper(gym.make('CartPole-v0'))
	state = env.reset()
	total_reward = 0.0
	steps = 0

	while True:
		state,reward,done,info = env.step(0)
		total_reward += 1
		steps +=1 
		if done:
			break
	print(f'Done at {steps} and Reward is {total_reward}')


