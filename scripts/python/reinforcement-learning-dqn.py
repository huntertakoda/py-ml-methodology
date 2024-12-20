import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import random

# initialize the CartPole environment

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# hyperparameters

gamma = 0.95  
epsilon = 1.0  
epsilon_decay = 0.995  
epsilon_min = 0.01  
learning_rate = 0.001
batch_size = 64
memory_size = 2000
episodes = 500

# replay memory for experience replay

memory = deque(maxlen=memory_size)

# build the Q-network

def build_model():
    model = Sequential([
        Dense(24, activation='relu', input_shape=(state_size,)),
        Dense(24, activation='relu'),
        Dense(action_size, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

# initialize the model

model = build_model()
target_model = build_model()  
target_model.set_weights(model.get_weights())

# optimized replay function

def replay():
    global epsilon
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)

    # separate minibatch into components
    
    states = np.array([m[0] for m in minibatch]).squeeze()
    actions = np.array([m[1] for m in minibatch])
    rewards = np.array([m[2] for m in minibatch])
    next_states = np.array([m[3] for m in minibatch]).squeeze()
    dones = np.array([m[4] for m in minibatch])

    # predict Q-values for current and next states
    
    current_qs = model.predict(states)
    next_qs = target_model.predict(next_states)

    # update Q-values with Bellman equation
    
    for i in range(batch_size):
        target = rewards[i]
        if not dones[i]:
            target += gamma * np.max(next_qs[i])
        current_qs[i][actions[i]] = target

    # train the model on the entire minibatch
    
    model.fit(states, current_qs, epochs=1, verbose=0, batch_size=batch_size)

    # reduce exploration rate
    
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# training loop

for episode in range(episodes):
    state, _ = env.reset()  
    state = state.reshape(1, state_size)
    total_reward = 0
    for time in range(500):
        
        # env.render()  # uncomment to visualize
        
        if np.random.rand() <= epsilon:
            action = np.random.choice(action_size) 
        else:
            action = np.argmax(model.predict(state)[0]) 
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = next_state.reshape(1, state_size)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        if done:
            print(f"Episode: {episode + 1}/{episodes}, Score: {total_reward}, Epsilon: {epsilon:.2f}")
            break
    replay()
    if episode % 10 == 0:
        target_model.set_weights(model.get_weights())

# save the trained model

model.save('dqn_cartpole_model.h5')
print("Model saved as 'dqn_cartpole_model.h5'")

