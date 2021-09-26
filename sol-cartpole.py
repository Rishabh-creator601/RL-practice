import gym


env = gym.make('CartPole-v0')
env.reset()


for i in range(1000):
	r = 0
	env.render()
	next_step,reward,done,info = env.step(env.action_space.sample())
	r += reward 
	if done:
		print(f'it takes {i} episodes to done ') #21 
		break
	if i % 10 == 0:
		print(f'Episode {i} reward : {reward}')

