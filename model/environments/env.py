import gym
import minigrid
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, RGBImgObsWrapper, FullyObsWrapper
from minigrid.envs.crossing import CrossingEnv

env_names = {
    'lava crossing': 'MiniGrid-LavaCrossingS11N5-v0',
    'simple crossing': 'MiniGrid-SimpleCrossingS11N5-v0',
    'dynamic obstacles': 'MiniGrid-Dynamic-Obstacles-16x16-v0',
    'door key': 'MiniGrid-DoorKey-5x5-v0',
    'empty': 'MiniGrid-Empty-16x16-v0'
}

import math
class ActionBonus(gym.Wrapper):
    """
    Wrapper which adds an exploration bonus.
    This is a reward to encourage exploration of less
    visited (state,action) pairs.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        env = self.unwrapped
        tup = (tuple(env.agent_pos), env.agent_dir, action)

        # Get the count for this (s,a) pair
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this (s,a) pair
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus
        
        #just for taking a step
        reward += -0.1

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.counts.clear()
        return self.env.reset(**kwargs)


class MiniGrid:
    #full_observability: False for partial or True for full
    
    def __init__(self, name, full_observability = False, action_bonus=False):
        
        self.full_observability = full_observability
        self.env = gym.make(env_names[name])
        
        if full_observability:
            self.env = FullyObsWrapper(self.env) # Get rid of the 'mission' field
        
        if action_bonus:
            self.env = ActionBonus(self.env)
         
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info
    
    def reset(self):
        obs, _ = self.env.reset()
        return obs
    
    def render(self):
        if self.full_observability:
            return self.env.get_full_render(False, 64)
        else:
            return self.env.get_frame(False, 64, True)