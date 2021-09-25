import numpy as np 


MATRIX_SIZE = 11 
goal = 10
M = np.matrix(np.ones(shape =(MATRIX_SIZE, MATRIX_SIZE)))
M *= -1


edges = [(0, 1), (1, 5), (5, 6), (5, 4), (1, 2), 
         (1, 3), (9, 10), (2, 4), (0, 6), (6, 7),
         (8, 9), (7, 8), (1, 7), (3, 9)]

for point in edges:
	if point[1] == goal:
		M[point] = 100
	else:
		M[point] = 0

	if point[0] == goal:
		M[point[::-1]] = 100
	else:
		M[point[::-1]]= 0
		# reverse of point

M[goal, goal]= 100
print(M)
# add goal point round trip




gamma = 0.8
initial_state=1

Q = np.matrix(np.zeros([MATRIX_SIZE,MATRIX_SIZE]))

def available_actions(state):
	current_state= M[state,]
	av_act = np.where(current_state >= 0)[1]
	return av_act


available_action = available_actions(initial_state)

def sample_next_step(available_actions_range):
	next_action = int(np.random.choice(available_action,1))
	return next_action


action = sample_next_step(available_action)

def update_index(current_state,action,gamma):
	max_index = np.where(Q[action,]==max(Q[action,]))[1]
	if max_index.shape[0] > 1:
		max_index = int(np.random.choice(max_index,size=1))
	else:
		max_index = int(max_index)
	
	max_value = Q[action,max_index]
	Q[current_state,action] = M[current_state,action] +gamma *max_value

	if (np.max(Q) > 1):
		return (np.sum(Q/np.max(Q)*100))
	else:
		return (0)


update_index(initial_state,action,gamma)


scores = []
for i in range(1000):
	current_state = np.random.randint(0,int(Q.shape[0]))
	
	available_act = available_actions(current_state)

	action = sample_next_step(available_act)
	score = update_index(current_state,action,gamma)
	scores.append(score)


current_state = 1
steps = [current_state]

while current_state !=10:
	next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]
	if next_step_index.shape[0] > 1:
		next_step_index = int(np.random.choice(next_step_index,size=1))
	else:
		next_step_index = int(next_step_index)
	steps.append(next_step_index)
	current_state = next_step_index


print(f'SELECTED PATH : {steps}')


