import gymnasium as gym
import matplotlib.pyplot as plt
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

env = gym.make('MiniGrid-Empty-8x8-v0')
env = RGBImgPartialObsWrapper(env) # Get pixel observations
env = ImgObsWrapper(env) # Get rid of the 'mission' field
obs, _ = env.reset() # This now produces an RGB tensor only

plt.imshow(obs)
plt.show()