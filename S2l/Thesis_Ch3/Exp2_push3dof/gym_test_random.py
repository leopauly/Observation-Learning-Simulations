#### Random agent in gym env

##Imports
import gym
import matplotlib.pyplot as plt

## Defining env
env = gym.make('Pusher3DOFReal-v1') 
env.switch=1
env.initialize_env()
#env = gym.make('Pusher7DOF-v1')
print(env.observation_space.shape[0])
print(env.action_space.shape[0])

## Defining vars
LR = 1e-3
goal_steps = 500
score_requirement = 50
initial_games = 10000


def some_random_games_first():
    # Each of these is its own game.
    for episode in range(200):
        env.reset()
        env.render(mode='human')
        img=env.render(mode='rgb_array')   # Get the observation
        #plt.imshow(img)
        #plt.show()
       

        # this is each frame, up to 200...but we wont make it that far.
        for t in range(200):
            
            # This will just create a sample action in any environment.
            # In this environment, the action can be 0 or 1, which is left or right
            action = env.action_space.sample()
            # This will display the environment
            # Only display if you really want to see it.
            # Takes much longer to display it.
            env.render(mode='human')
            img=env.render(mode='human')   # Get the observation
            
            #plt.imshow(img)
            #plt.show()
            

            #print(episode,t)


            # this executes the environment with an action, 
            # and returns the observation of the environment, 
            # the reward, if the env is over, and other info.
            observation, reward, done, info = env.step([100,0,0])
	     
		
            env.render(mode='human')
            observation, reward, done, info = env.step([-100,0,0])
	    #print(env.get_eval())
            #if done:
            #    break
                
some_random_games_first()
