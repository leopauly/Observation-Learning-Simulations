#### Random agent in gym env

##Imports
import gym
import matplotlib.pyplot as plt

## Defining env
env = gym.make('LunarLanderContinuous-v2') 
#env = gym.make('Pusher7DOF-v1')
print(env.observation_space.shape[0])
print(env.action_space.shape[0])

## Defining vars
LR = 1e-3
goal_steps = 500
score_requirement = 50
initial_games = 10000


def some_random_games_first():
    for episode in range(200):
        env.reset()
        env.render(mode='human')
        img=env.render(mode='rgb_array')  
        import time
        time.sleep(2000) 
        
       
        for t in range(200):
            
            env.render(mode='human')
            img=env.render(mode='human')  
            observation, reward, done, info = env.step(1,-1)
          
                
some_random_games_first()
