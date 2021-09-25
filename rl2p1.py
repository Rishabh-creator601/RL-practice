import gym
import warnings
warnings.filterwarnings('ignore')


print('imported')

env = gym.make('CartPole-v0')
env.reset()


print('ENV GENERATED ')

for i in range(100):
	print(f'{i} out of 100 done !')
	env.render()
	env.step(env.action_space.sample())
	