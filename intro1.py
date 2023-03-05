import gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v2")

env.reset()

# some info you can get
print("a sample of actions: ", env.action_space.sample())
print("observation space shape: ", env.observation_space.shape)
print("a sample of observation space: ", env.observation_space.sample())

# # this is an example of the model working on 200 steps and printing all the rewards
# for step in range(200):
#     env.render()
#     obs, reward, done, info = env.step(env.action_space.sample()) # perform a random action
#     print(reward)

TIMESTEPS = 10000

# let's try to use the RL algorithm A2C
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=TIMESTEPS)

episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        obs, reward, done, info = env.step(env.action_space.sample())  # perform a random action

# notice that it's very easy to change the algorithm. Just import the one that you like and change the line with the
# model definition
env.close()