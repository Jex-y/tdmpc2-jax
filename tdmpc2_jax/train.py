import gymnasium as gym
import numpy as np
import jax
import flax.linen as nn
import tqdm
from networks import NormedLinear
from common.activations import mish, simnorm
from functools import partial
from world_model import WorldModel
from tdmpc2 import TDMPC2
from data import EpisodicReplayBuffer
import os
import hydra
import jax.numpy as jnp
from wrappers.action_scale import RescaleActions
import wandb
from gymnasium.core import ActType, ObsType
import time
from typing import Any, Dict, List, SupportsFloat, Tuple
import omegaconf


os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

class CustomMonitor(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.

    :param env: The environment
    :param filename: the location to save a log file, can be None for no log
    """

    def __init__(
        self,
        env: gym.Env,
        log_dir: str = None,
        record_freq: int = 5,
        no_improvement_window: int = 100,
    ):
        super().__init__(env=env)

        self.t_start = time.time()
        self.results_writer = None
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        self.log_file_path = os.path.join(log_dir, "agent-log.txt")
        self.log_file = open(self.log_file_path, "w")

        self.rewards: List[float] = []
        self.episode_rewards: List[float] = []
        self.needs_reset = True
        self.episodes = 0
        self.cumulative_reward = 0.0
        self.record_freq = record_freq
        self.recording = False
        self.video_frames = []
        self.ep_since_improvement = 0
        self.best_reward = -np.inf
        self.no_improvement_window = no_improvement_window

    def reset(self, **kwargs) -> Tuple[ObsType, Dict[str, Any]]:
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
        episode_reward = sum(self.rewards)
        episode_length = len(self.rewards)

        if episode_length > 0:
            self.episodes += 1
            self.cumulative_reward += episode_reward
            self.episode_rewards.append(episode_reward)
            self.log_file.write(f"episode: {self.episodes}, reward: {episode_reward}\n")
            self.log_file.flush()

            log_data = {
                "episode_reward": episode_reward,
                "episode_length": episode_length,
                "cumulative_reward": self.cumulative_reward,
                "average_reward (last 10 episodes)": np.mean(
                    self.episode_rewards[-10:]
                ),
                "average_reward (last 100 episodes)": np.mean(
                    self.episode_rewards[-100:]
                ),
            }

            if self.recording and len(self.video_frames) > 0:
                video_array = np.stack(self.video_frames)
                # (time, height, width, channel) -> (time, channel, height, width)
                video_array = np.transpose(video_array, (0, 3, 1, 2))

                video = wandb.Video(
                    video_array,
                    caption=f"Episode {self.episodes}, Reward: {episode_reward:.0f}",
                    fps=30,
                )

                log_data["video"] = video
                self.video_frames = []

            wandb.log(log_data, step=self.episodes)

            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.ep_since_improvement = 0
            else:
                self.ep_since_improvement += 1

                assert (
                    self.episodes < 2 * self.no_improvement_window
                    or sum(self.episode_rewards[-self.no_improvement_window :])
                    > sum(
                        self.episode_rewards[
                            -2
                            * self.no_improvement_window : -self.no_improvement_window
                        ]
                    )
                ), f"Mean reward did not improve for {self.no_improvement_window} episodes"

        self.rewards = []
        self.needs_reset = False
        self.recording = self.record_freq > 0 and self.episodes % self.record_freq == 0

        return self.env.reset(**kwargs)

    def step(
        self, action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")

        observation, reward, terminated, truncated, info = self.env.step(action)
        self.rewards.append(float(reward))

        if self.recording:
            self.video_frames.append(self.env.render())

        if terminated or truncated:
            self.needs_reset = True

        return observation, reward, terminated, truncated, info

    def __del__(self):
        self.log_file.close()
        # upload the log file to wandb
        wandb.save(self.log_file_path)


@hydra.main(config_name='config', config_path='.', version_base='1.2')
def train(cfg: dict):
  seed = 42
  max_episodes = cfg['max_episodes']
  encoder_config = cfg['encoder']
  model_config = cfg['world_model']
  tdmpc_config = cfg['tdmpc2']
  seed_steps = cfg['seed_steps']
  buffer_size = cfg['buffer_size']
  no_improvement_window = cfg['no_improvement_window']
  
  wandb.config = omegaconf.OmegaConf.to_container(cfg)
  wandb.init(project='tdmpc2-jax', entity='ejex', config=dict(cfg))

  env = gym.make("BipedalWalker-v3", render_mode='rgb_array', hardcore=cfg['hardcore'])
  
  env = RescaleActions(env)
  env = CustomMonitor(env, log_dir="logs", record_freq=0, no_improvement_window=no_improvement_window)  
  
  env.action_space.seed(seed)
  env.observation_space.seed(seed)
  np.random.seed(seed)
  rng = jax.random.PRNGKey(seed)

  dtype = jnp.dtype(model_config['dtype'])
  rng, model_key = jax.random.split(rng, 2)
  encoder = nn.Sequential(
      [
          NormedLinear(encoder_config['encoder_dim'],
                       activation=mish, dtype=dtype)
          for _ in range(encoder_config['num_encoder_layers']-1)
      ] +
      [
          NormedLinear(
              model_config['latent_dim'],
              activation=partial(
                  simnorm, simplex_dim=model_config['simnorm_dim']),
              dtype=dtype)
      ])

  model = WorldModel.create(
      observation_space=env.observation_space,
      action_space=env.action_space,
      encoder_module=encoder,
      **model_config,
      key=model_key)
  agent = TDMPC2.create(world_model=model, **tdmpc_config)

  replay_buffer = EpisodicReplayBuffer(
      capacity=buffer_size,
      dummy_input=dict(
          observation=env.observation_space.sample(),
          action=env.action_space.sample(),
          reward=1.0,
          next_observation=env.observation_space.sample(),
          terminated=True,
          truncated=True,
      ),
      seed=seed,
      respect_episode_boundaries=False)

  # Training loop
  prev_plan = None
  observation, _ = env.reset(seed=seed)
  done = False
  step_count = 0
  
  for ep_count in tqdm.tqdm(range(max_episodes), smoothing=0.1):
    while not done:
        if step_count <= seed_steps:
            action = env.action_space.sample()
        else:
            rng, action_key = jax.random.split(rng)
            
            action, prev_plan = agent.act(
                observation, prev_plan, train=True, key=action_key)
            
        next_observation, reward, terminated, truncated, _ = env.step(action)
        
        done = terminated or truncated
        step_count += 1
        
        replay_buffer.insert(dict(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            terminated=terminated,
            truncated=truncated),
            episode_index=ep_count)
        
        observation = next_observation

        if step_count >= seed_steps:
            if step_count == seed_steps:
                print('Pre-training on seed data...')
                num_updates = seed_steps
            else:
                num_updates = 1

            rng, *update_keys = jax.random.split(rng, num_updates+1)
            for j in range(num_updates):
                batch = replay_buffer.sample(agent.batch_size, agent.horizon)

                agent, train_info = agent.update(
                    observations=batch['observation'],
                    actions=batch['action'],
                    rewards=batch['reward'],
                    next_observations=batch['next_observation'],
                    terminated=batch['terminated'],
                    truncated=batch['truncated'],
                    key=update_keys[j])
    
    observation, _ = env.reset()
    prev_plan = None
    done = False

if __name__ == '__main__':
  train()
