#!/usr/bin/env python
# coding: utf-8

# In[2]:


import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")

learning_rate = 0.1
discount = 0.95
episodes = 250000
show_every = 500

discrete_os_size = [20]*len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high-env.observation_space.low)/discrete_os_size

epsilon = 0.5
start_epsilon_decaying = 1
end_epsilon_decaying = episodes//2

epsilon_decay_value = epsilon/(end_epsilon_decaying-start_epsilon_decaying)
 
qtable = np.random.uniform(low=-2,high=0,size=(discrete_os_size+[env.action_space.n]))

ep_rewards = []
aggr_ep_rewards = {'ep':[],'avg':[],'min':[],'max':[]}

def get_discrete_state(state):
    discrete_state = (state-env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

for episode in range(episodes):
    episode_reward = 0
    if episode%show_every==0:
        print(episode)
        render = True
    else:
        render = False
    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:
        if np.random.random()>epsilon:
            action = np.argmax(qtable[discrete_state])
        else:
            action = np.random.randint(0,env.action_space.n)
        action = np.argmax(qtable[discrete_state])
        new_state,reward,done,_ = env.step(action)
        episode_reward+=reward
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()

        if not done:
            max_future_q = np.max(qtable[new_discrete_state])
            current_q = qtable[discrete_state+(action,)]
            new_q = (1-learning_rate)*current_q+learning_rate*(reward+discount*max_future_q)
            qtable[discrete_state+(action,)] = new_q

        elif new_state[0]>=env.goal_position:
            print(f"we made it on episode {episode}")
            qtable[discrete_state+(action,)] = 0

        discrete_state = new_discrete_state
    if end_epsilon_decaying >=episode>=start_epsilon_decaying:
        epsilon -= epsilon_decay_value
    ep_rewards.append(episode_reward)
    
    if not episode%show_every:
        average_rewards = sum(ep_rewards[-show_every:])/len(ep_rewards[-show_every:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_rewards)
        aggr_ep_rewards['min'].append(min(ep_rewards[-show_every:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-show_every:]))
        
        print(f"episode: {episode} avg:{average_rewards} min:{min(ep_rewards[-show_every:])} max:{max(ep_rewards[-show_every:])}")
env.close()

plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['avg'],label='avg')
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['min'],label='min')
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['max'],label='max')
plt.legend(loc=4)
plt.show()


# In[ ]:




