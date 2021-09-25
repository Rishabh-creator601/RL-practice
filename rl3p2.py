import gym 


env = gym.make('BipedalWalker-v3')
env.reset()

observations =  []
dones= []
infos = []
rewards = []



def generate_logs(name,cont):
	with open(f'{name}.txt','w') as f:
		f.write(f'{cont}')
	f.close()

for i in range(1000):
	env.render()
	action  =env.action_space.sample()
	observation,reward,done,info = env.step(action)
	observations.append(observation)
	rewards.append(reward)
	dones.append(done)
	infos.append(info)
	print(f'{i} of 1000')



obs = [a for a in observations]
rew = [b for b in rewards]
inf = [c for c in infos]


generate_logs('obs',obs)
generate_logs('rewards',rew)
generate_logs('info',inf)

print('done')


