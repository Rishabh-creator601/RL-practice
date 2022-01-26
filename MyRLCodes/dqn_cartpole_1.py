import math,gym,random
import numpy as np 
import tensorflow.compat.v1 as tf


tf.disable_v2_behavior() 


print("Imported ")


env = gym.make("CartPole-v0")



input_shape = len(env.reset())
hidden_1 = 64 
output_shape = env.action_space.n 


gamma = 0.99 
alpha = 0.85 
epsilon = 0.8 
num_episodes = 100 
lr = 0.001




i1 = tf.placeholder(shape=[1,input_shape],dtype=tf.float32)
w1 = tf.Variable(tf.random_normal([input_shape,hidden_1]))
w_output  = tf.Variable(tf.random_normal([hidden_1,output_shape]))



layer_1 = tf.matmul(i1,w1)
layer_1 = tf.nn.relu(layer_1)

layer_output = tf.matmul(layer_1,w_output)

best_action = tf.argmax(layer_output,1)



# TRAINING PHASE 



next_q = tf.placeholder(shape=[1,output_shape],dtype=tf.float32)
loss = tf.reduce_mean(tf.square(next_q - layer_output))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)



init = tf.global_variables_initializer()

saver = tf.train.Saver()


with tf.Session() as sess:

	sess.run(init)


	reward_list = []

	for i in range(num_episodes):

		done = False
		state = np.reshape(env.reset(),[1,input_shape])

		reward_ = 0

		while not done:
			action,q_vals = sess.run([best_action,layer_output],{i1:state})


			if np.random.rand(1)  < epsilon:
				action[0] = env.action_space.sample()

			next_state,reward,done,info =  env.step(action[0])

			next_state = np.reshape(next_state,[1,input_shape])

			max_q = np.max(sess.run([w_output],feed_dict={i1:next_state}))

			target_q = q_vals

			target_q[0,action[0]] =  reward + gamma * max_q
			_,_,curr_loss = sess.run([optimizer,w_output,loss],feed_dict={i1:state,next_q:target_q})

			reward_ += reward 



		epsilon =  0.01  + (epsilon - 0.01 ) * math.exp(-0.001 * i)

		reward_list.append(reward_)

		past_three = np.mean(reward_list[-3:])

		if past_three > 190:
			print("Solved ")
			saver.save(sess,"/tmp/model.ckpt")
			break 

		saver.save(sess,"/tmp/model.ckpt")







# TESTING PHASE 

total_reward = 0.0

saver = tf.train.Saver()

with tf.Session() as sess:
	saver.restore(sess,"/tmp/model.ckpt")

	state = env.reset()

	for i in range(1000):

		done = False

		while not done:

			state = np.reshape(state,[1,input_shape])

			action = sess.run(best_action,feed_dict={i1:state})
			state,reward,done,_ =  env.step(action[0])

			total_reward += reward 

			env.render()

		print(f"Epiode : {i} Reward : {reward}")


		env.reset()



print("Average Reward Per Episode : {}".format(total_reward/1000))



