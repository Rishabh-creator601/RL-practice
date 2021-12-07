import random , gym 
import numpy as np 
from collections import deque 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.optimizers import Adam 

import time as t


EPISODES = 1000 


class DQNAgent:

	def __init__(self,state_size,action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen=2000)
		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.01
		self.model = self.get_model()


	def get_model(self):
		model = Sequential([
             Dense(24,input_dim=self.state_size,activation='relu'),
             Dense(24,activation='relu'),
             Dense(self.action_size,activation='relu')

		])

		model.compile(loss='mse',optimizer=Adam(learning_rate=self.learning_rate))

		return model 


	def memorize(self,state,action,next_action,reward):
		self.memory.append((state,action,next_action,reward))

	def act(self,state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)

		values = self.model.predict(state)
		return np.argmax(values[0])

	def log(self,cont):
		with open('Log.txt','w')as f:
			f.write(cont)
		f.close()

	def replay(self,batch_size):
		minibatch = random.sample(self.memory,batch_size)
		self.log(minibatch)


		for state, action,reward, next_state,done in minibatch:
			target = reward 

			if not done:
				target = (reward + self.gamma * 
					np.amax(self.model.predict(next_state[0])))
			target_f = self.model.predict(state)
			target_f[0][action] = target 

			self.model.fit(state,target_f,epoch=1)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay


	def load(self,name):
		self.model.load_weights(name)

	def save(self,name):
		self.model.save_weights(name)

if __name__ == '__main__':
	env = gym.make('CartPole-v0')
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n 

	agent = DQNAgent(state_size,action_size)
	total_scores =[]


	done = False 
	batch_size = 32

	for i in range(EPISODES):
		state = env.reset()
		state =np.reshape(state, [1,state_size])

		for time in range(500):
			action = agent.act(state)

			next_action, reward , done , _ = env.step(action)

			reward = reward if not done else -10

			total_scores.append(reward)

			next_action = np.reshape(next_action,[1,state_size])

			if done:

				print("Episode : {}/{} score : {} e : {:.2}".format(i,EPISODES, time,agent.epsilon))

				break 
			if len(agent.memory) > batch_size:
				agent.replay(batch_size)
		if i % 10 == 0:
			agent.save('cartpole_dqn.h5')
	print('Average Reward :',np.mean(total_scores))

		





