import gym
from stable_baselines3 import A2C
import os

models_dir = "models/A2C"
logs_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

env = gym.make("LunarLander-v2")
env.reset()

TIMESTEPS = 10000

# let's try to use the RL algorithm A2C
model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir)

for i in range(1,10):
    # This allows us to see the actual total number of timesteps for the model rather than resetting every iteration
    # If you specify different tb_log_name in subsequent runs, you will have split graphs
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C")
    # save the model in the previous dir
    model.save(f"{models_dir}/{TIMESTEPS * i}")

# in the terminal you can type "tensorboard --logdir=logs" (after having installed tensorboard -> pip install tensorboard)
# then click on the link
# to compare algorithms, you should train models for both algorithms. So, I simply used "PPO" and "A2C" algorithms (replace)
# notice that tensorboard is online, so you can see the statistics while it is still looping


# notice that it's very easy to change the algorithm. Just import the one that you like and change the line with the
# model definition
env.close()