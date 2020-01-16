#### Random agent in gym env

##Imports
import gym
import matplotlib.pyplot as plt

## Defining env
env = gym.make('Pusher3DOFReal-v1') #('Pusher7DOF-v1')    #('Pusher-v1') 
print(env.observation_space.shape[0])
print(env.action_space.shape[0])

## Defining vars
LR = 1e-3
goal_steps = 500
score_requirement = 50
initial_games = 10000

## Function for running the Env
def some_random_games_first():
    for episode in range(600):
        env.reset()
        env.render(mode='human')
        img=env.render(mode='rgb_array')   # Get the observation
        #plt.imshow(img)
        #plt.show()
       

        # this is each frame, up to 200...but we wont make it that far.
        for t in range(200):
            
            
            action = env.action_space.sample()

            
            print('Distance to goal:',env.get_eval())
          
            observation, reward, done, info = env.step([10,0,0,0])
            env.render(mode='human')
            #observation, reward, done, info = env.step([3,0,0,0])
            #env.render(mode='human')
            #img=env.render(mode='human')  
          
                
some_random_games_first()
