import numpy as np
import torch
import argparse
import time
import os
import yaml
from tqdm import tqdm
from utils.logger import Logger, EvalMonitor
from utils.train_utils import initiate_class
from utils.video import VideoRecorder

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, logger, monitor, video_recorder, seed, env_cfg, step, nb_eval_episodes):
    eval_env = initiate_class(env_cfg['name'], env_cfg['settings'])
    eval_env.seed(seed+step)
    average_episode_reward, list_episode_reward = 0, []
    
    for episode in tqdm(range(0, nb_eval_episodes), desc ="Model Evaluation"):
        obs, info = eval_env.reset()
        video_recorder.init(enabled=(episode == 0))
        done = False
        episode_reward, episode_step = 0, 0

        while not done and (episode_step < env_cfg['settings']['max_it']):
            action = policy.act(obs, sample=False)
            obs, reward, done, info = eval_env.step(action)
            video_recorder.record(eval_env)
            episode_reward += reward
            episode_step += 1

        average_episode_reward += episode_reward
        list_episode_reward.append(episode_reward)
        video_recorder.save(f'{step}.mp4')
        monitor.collect_episode_performance(info)

    average_episode_reward /= nb_eval_episodes
    logger.log('eval/episode_reward', average_episode_reward, step)
    logger.log_histogram('eval/episode_reward_distribution', np.array(list_episode_reward), step)
    logger.dump(step)
    monitor.dump_end_eval(logger, step)


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
    video_recorder = VideoRecorder(exp_folder + "videos/") #self.work_dir if cfg.save_video else None)
    eval_monitor = EvalMonitor(cfg['logger'])
    
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
    if 'load' in policy_cfg.keys():
        policy.load(policy_cfg['load'])

    # Buffer initialization and data collection
    memory_cfg = cfg['memory']
    memory_cfg['settings']['obs_dim'] = env.observation_space.shape[0]
    memory_cfg['settings']['action_dim'] = env.action_space.shape[0]
    memory = initiate_class(memory_cfg['name'], memory_cfg['settings'], multiple_arguments=False)
    priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)
    
    data_collection_noise(env=env, 
                          memory=memory, 
                          data_collection_steps=cfg['train']['data_collection_steps'],
                          reset_noise_steps=cfg['train']['replay_frequency'],
                          reward_clip=cfg['train']['reward_clip'])

    step, episode, episode_reward, done = 0, 0, 0, True
    start_time = time.time()
    for step in range(cfg['train']['train_steps']):    
        # log end of episode or truncation and reset env
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
        action = policy.act(obs)

        next_obs, reward, done, _ = env.step(action)

        episode_reward += reward
        if args.reward_clip > 0:
            reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
        memory.add(obs, action, reward, done)

        obs = next_obs
        episode_step += 1

        # Train
        memory.priority_weight = min(memory.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1
        if step % cfg["replay_frequency"] == 0:
            policy.reset_noise()
            policy.update(memory, logger, step)
        # Update target network
        if step % cfg["target_update"] == 0:
            policy.update_target_net(logger, step)

        # Test
        if step % cfg['train']['eval_frequency'] == 0:
            logger.log('eval/episode', episode, step)
            eval_policy(policy, logger, eval_monitor, video_recorder, ENV_SEED, env_cfg, step, cfg['train']['nb_eval_episodes'])

        if step % cfg['checkpoint_frequency'] == 0:
            policy.save(exp_folder+"checkpoint/", f"checkpoint_{step}.ph")
        
    policy.save(exp_folder+"models/")

def data_collection_noise(env, 
                          memory, 
                          policy, 
                          data_collection_steps, 
                          reset_noise_steps,
                          reward_clip):
    done = True

    for step in tqdm(range(0, data_collection_steps), desc ="Replay Buffer filling"):
        if done:
            obs, _ = env.reset()
            done = False

        if step % reset_noise_steps==0:
            policy.reset_noise()

        # sample action for data collection
        action = policy.act(obs)
        _, reward, done, _ = env.step(action)
        if reward_clip > 0:
            reward = max(min(reward, args.reward_clip), -args.reward_clip) 
        memory.append(obs, action, reward, done)

def data_collection(env, replay_buffer, data_collection_steps):
    done = True

    for step in tqdm(range(0, data_collection_steps), desc ="Replay Buffer filling"):
        if done:
            obs, _ = env.reset()
            done = False

        # sample action for data collection
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        replay_buffer.add(obs, action, next_obs, reward, done)
        obs = next_obs 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path")
    args = parser.parse_args()
    run(args)