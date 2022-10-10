import string
import numpy as np
import torch
import argparse
import time
import os
import sys
import yaml
from tqdm import tqdm
from utils.logger import Logger
from utils.train_utils import initiate_class
from utils.video import VideoRecorder

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, logger, video_recorder, seed, env_cfg, step, nb_eval_episodes):
    eval_env = initiate_class(env_cfg['name'], env_cfg['settings'])
    eval_env.seed(seed+step)
    average_episode_reward = 0
    for episode in range(nb_eval_episodes):
        obs, _ = eval_env.reset()
        video_recorder.init(enabled=(episode == 0))
        done = False
        episode_reward, episode_step = 0, 0
        while not done and (episode_step < env_cfg['settings']['max_it']):
            action = policy.act(obs, sample=False)
            obs, reward, done, _ = eval_env.step(action)
            video_recorder.record(eval_env)
            episode_reward += reward
            episode_step+=1

        average_episode_reward += episode_reward
        video_recorder.save(f'{step}.mp4')
    average_episode_reward /= nb_eval_episodes
    logger.log('eval/episode_reward', average_episode_reward, step)
    logger.dump(step)


def run(args):
    with open(args.cfg_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    today_date = time.strftime("%Y%m%d")
    today_time = time.strftime("%H%M%S")
    start_time = time.time()
    exp_folder = f"./experiments/{today_date}/{today_time}_{cfg['experiment']}/"
    os.makedirs(exp_folder+"models/", exist_ok=True)
    os.makedirs(exp_folder+"videos/", exist_ok=True)
    os.makedirs(exp_folder+"checkpoint/", exist_ok=True)

    # Save experiment configuration
    with open(f'{exp_folder}config.yaml', 'w') as file:
        documents = yaml.dump(cfg, file)
    
    # Logger
    logger = Logger(exp_folder,
                    save_tb=cfg['train']['save_tensorboard'],
                    log_frequency=cfg['train']['log_frequency'],
                    agent=cfg['agent'])
    video_recorder = VideoRecorder(None) #self.work_dir if cfg.save_video else None)

    # Env init
    env_cfg = cfg['env']
    env = initiate_class(env_cfg['name'], env_cfg['settings'])

    # Set seeds
    ENV_SEED = env_cfg['settings']['seed']
    env.seed(ENV_SEED)
    env.action_space.seed(ENV_SEED)
    torch.manual_seed(ENV_SEED)
    np.random.seed(ENV_SEED)

    # Observations and actions space shapes
    policy_cfg = cfg['policy']
    policy_cfg['settings']['obs_dim'] = env.observation_space.shape[0]
    policy_cfg['settings']['action_dim'] = env.action_space.shape[0]

	# Initialize policies
    policy = initiate_class(cfg['policy']['name'], policy_cfg, multiple_arguments=False)
    
    # Buffer initialization and data collection
    buffer_cfg = cfg['buffer']
    buffer_cfg['settings']['obs_dim'] = env.observation_space.shape[0]
    buffer_cfg['settings']['action_dim'] = env.action_space.shape[0]
    replay_buffer = initiate_class(buffer_cfg['name'], buffer_cfg['settings'])
    data_collection(env, replay_buffer, cfg['train']['data_collection_steps'])
    reward_processor = initiate_class(cfg['train']['reward processor']['name'], cfg['train']['reward processor']['settings'])
    reward_processor.fit(replay_buffer.reward)

    step, episode, episode_reward, done = 0, 0, 0, True
    start_time = time.time()
    while step < cfg['train']['train_steps']:
        # evaluate agent periodically
        if step % cfg['train']['eval_frequency'] == 0:
            logger.log('eval/episode', episode, step)
            eval_policy(policy, logger, video_recorder, ENV_SEED, env_cfg, step, cfg['train']['nb_eval_episodes'])

        if done or (episode_step >= cfg['train']['max_episode_steps']):
            if step > 0:
                logger.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                logger.dump(step)

            logger.log('train/episode_reward', episode_reward, step)

            obs, _ = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

            logger.log('train/episode', episode, step)

        # sample action for data collection
        action = policy.act(obs, sample=True)

        # run training update
        policy.update(replay_buffer, reward_processor, logger, step)

        next_obs, reward, done, _ = env.step(action)
        reward_processor.update(reward)

        # allow infinite bootstrap
        done = float(done)
        episode_reward += reward

        replay_buffer.add(obs, action, next_obs, reward, done)

        obs = next_obs
        episode_step += 1
        step += 1


def data_collection(env, replay_buffer, data_collection_steps):
    done = True

    #for step in range(data_collection_steps):
    for step in tqdm(range(0, data_collection_steps), desc ="Replay Buffer filling"):
        if done:
            obs, _ = env.reset()
            done = False

        # sample action for data collection
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        replay_buffer.add(obs, action, next_obs, reward, done)

        obs = next_obs
        step += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path")
    args = parser.parse_args()
    run(args)