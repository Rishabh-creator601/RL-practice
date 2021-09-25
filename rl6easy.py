import gym
import warnings
warnings.filterwarnings('ignore')


env = gym.make('CartPole-v0')


for i in range(100):
	Return = 0
	state = env.reset()
	for j in range(50):
		env.render()
		random_action = env.action_space.sample()
		next_step,reward,done,info = env.step(random_action)
		Return = Return + reward
		if done:
			break

	if i % 10 == 0:
		print(f'Episode  : {i} Reward : {Return}')
env.close()