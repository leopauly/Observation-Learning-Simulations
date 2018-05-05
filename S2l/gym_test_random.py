#### Random agent in gym env

##Imports
import gym
import matplotlib.pyplot as plt

## Defining env
env = gym.make('Pusher3DOFReal-v1')
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
        plt.imshow(img)
        plt.show()
       

        # this is each frame, up to 200...but we wont make it that far.
        for t in range(10):
            
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
            

            print(episode,t)


            # this executes the environment with an action, 
            # and returns the observation of the environment, 
            # the reward, if the env is over, and other info.
            observation, reward, done, info = env.step(action)
            if done:
                break
                
some_random_games_first()
