from collections import deque 
import gym,random,os 
import numpy as np 


print("-"*25+"Imported Sector 1"+"-"*25)


from tensorflow.keras.layers import Dense 
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.optimizers import Adam 


print("-"*25+"Imported Sector 2"+"-"*25)



env  = gym.make("CartPole-v1")



state_size = env.observation_space.shape[0]
action_size = env.action_space.n 


num_episodes = 1001
timesteps = 5000 




class DQNAgent:

	def __init__(self,state_size,action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen=2000)
		self.gamma = 0.99 
		self.epsilon = 1.0
		self.batch_size = 32
		self.epsilon_min =0.1
		self.lr = 0.001
		self.epsilon_decay = 0.995 
		self.model = self.load("Saved_Models/dqn_cartpole.h5")


	def build(self):
		model = Sequential([
               Dense(24,input_dim=self.state_size,activation='relu'),
               Dense(24,activation='relu'),
               Dense(self.action_size,activation='linear')

		])

		model.compile(loss='mse',optimizer=Adam(learning_rate=self.lr))

		return model 


	def act(self,state):

		if np.random.rand() <= self.epsilon:
			return random.choice(np.arange(self.action_size))
		else:
			act_values = self.model.predict(state)
			return np.argmax(act_values[0])



	def remember(self,state,action,reward,next_state,done):
		self.memory.append((state,action,reward,next_state,done))

	def replay(self,batch_size):

		minibatch = random.sample(self.memory,batch_size)

		for state,action,reward,next_state,done in minibatch:
			target = reward 

			if not done:
				target =  (reward + self.gamma * np.amax(next_state[0]))

			target_f = self.model.predict(state)

			target_f[0][action] = target 

			self.model.fit(state,target_f,epochs=1,verbose=0)

		if self.epsilon >= self.epsilon_min:
			self.epsilon *= self.epsilon_decay


	def load(self,name):
		return load_model(name)

	def save(self,name):
		self.model.save(name)





agent = DQNAgent(state_size,action_size)

print("Initializing Agent  :",agent)

done = False


print("-"*25+"Training CartPole "+"-"*25)


for e in range(num_episodes):
	state = env.reset()
	state = np.reshape(state,[1,state_size])

	while not done:


		action  = agent.act(state)

		next_state,reward,done,info = env.step(action)

		reward = reward if not done else -10

		next_state = np.reshape(next_state,[1,state_size])

		agent.remember(state,action,reward,next_state,done)

	if done:
		print(f"Episodes :{e}  epsilon : {agent.epsilon}")


	if len(agent.memory) > agent.batch_size:
		agent.replay(agent.batch_size)


