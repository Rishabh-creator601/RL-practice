import gym ,random ,os 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.models import load_model,Sequential
from tensorflow.keras.optimizers import Adam
from collections import deque 
import numpy as np 






class DQNAgent:
	def __init__(self,state_size,action_size,env,model=None):
		self.env = env 
		self.state_size = state_size
		self.action_size = action_size
		self.gamma = 0.99
		self.epsilon = 1.0
		self.epsilon_min  = 0.1
		self.epsilon_decay = 0.995 
		self.memory =deque(maxlen=10000)
		self.lr = 0.001 
		self.batch_size = 64 

		if model is None:
			self.model = self.build_model()
		else:
			self.model = model 



	def build_model(self):
		model = Sequential([
              Dense(24,input_dim=self.state_size,activation='relu'),
              Dense(48,activation='relu'),
              Dense(self.action_size,activation='linear')

		])

		model.compile(loss='mse',optimizer=Adam(learning_rate=self.lr))

		return model 

	def remember(self,state,action,reward,next_state,done):
		self.memory.append((state,action,reward,next_state,done))

	def act(self,state):
		if np.random.randn() <= self.epsilon:
			return random.choice(np.arange(self.action_size))
		else:

			return np.argmax(self.model.predict(state)[0])

	def replay(self,batch_size):
		minibatch = random.sample(self.memory,batch_size)


		for state,action,reward,next_state,done in minibatch:

			target = reward

			if not done:
				target = (reward + self.gamma * np.amax(next_state[0]))



			target_f  = self.model.predict(state)

			target_f[0][action] = target


			self.model.fit(state,target_f,epochs=1,verbose=0)



		if self.epsilon >= self.epsilon_min:
			self.epsilon *= self.epsilon_decay


	def save(self,name):
		self.model.save(name)

	def load(self,name):
		return load_model(name)

	def save_w(self,name):
		self.model.save_weights(name)

	def load_w(self,name):
		self.model.load_weights(name)


	def sample_episode(self,n_iters=1001,render_every=100,time_steps=1000,render=False,save_threshold=100,log_value=50,model_name='model.h5'):

		done = False

		scores_ = deque(maxlen=100)  

		for e in range(n_iters):
			state = self.env.reset()
			state = np.reshape(state,[1,self.state_size])

			score =  0 
			

			for t in range(time_steps):





				action = self.act(state)

				next_state,reward,done,info = self.env.step(action)


				

				next_state =  np.reshape(next_state,[1,self.state_size])



				if next_state[0][0] >= 0.5:
					reward += 10 




				score += reward  

				self.remember(state,action,reward,next_state,done)


				state = next_state

				if done:
					break 



			scores_.append(score)

			if e % log_value == 0:

				print(f"Epiode : {e} Score :{t}  Average Score : {np.mean(scores_)}")


			if len(self.memory) > self.batch_size:
				self.replay(self.batch_size)

			if e % save_threshold == 0:
				self.save(model_name)
				print(f"Saving Model at {e}")


			
def check_dir(name):
	if not os.path.exists(name):
		os.mkdir(name)




