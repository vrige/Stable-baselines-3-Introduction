import gym
from stable_baselines3 import A2C
import os

models_dir = "models/A2C"
logs_dir = "logs"

env = gym.make("LunarLander-v2")
env.reset()

# load a model
model_path = f"{models_dir}/80000.zip"
model = A2C.load(model_path, env=env)

episodes = 5


for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        print(rewards)

env.close()