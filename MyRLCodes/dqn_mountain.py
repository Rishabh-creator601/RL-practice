from dqn import DQNAgent ,check_dir 
import gym , os 


check_dir("_saved_")







env = gym.make("MountainCar-v0")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n 




agent = DQNAgent(state_size,action_size,env)
agent.batch_size = 128 
agent.sample_episode(time_steps=5000,render=True,log_value=1,model_name='_saved_/mountainCar.h5')