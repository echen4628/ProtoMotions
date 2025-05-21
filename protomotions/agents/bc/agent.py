import torch
import os
import logging

from torch import Tensor

import time
import math
from pathlib import Path
from typing import Optional, Tuple, Dict

from lightning.fabric import Fabric

from hydra.utils import instantiate
from isaac_utils import torch_utils

from protomotions.utils.time_report import TimeReport
from protomotions.utils.average_meter import AverageMeter, TensorAverageMeterDict
from protomotions.agents.utils.data_utils import DictDataset, ExperienceBuffer
from protomotions.agents.ppo.model import PPOModel
from protomotions.agents.common.common import weight_init, get_params
from protomotions.envs.base_env.env import BaseEnv
from protomotions.utils.running_mean_std import RunningMeanStd
from rich.progress import track
from protomotions.agents.ppo.utils import discount_values, bounds_loss

log = logging.getLogger(__name__)


class BC:
    def __init__(self, fabric: Fabric, env: BaseEnv, config):
        self.fabric = fabric
        self.device: torch.device = fabric.device
        self.env = env
        self.motion_lib = self.env.motion_lib
        self.config = config
        self.experts = {}

        self.num_envs: int = self.env.config.num_envs
        self.num_steps: int = config.num_steps

        self.current_epoch = 0
        self.step_count = 0
        self.fit_start_time = None
        self.best_evaluated_score = None
        self.episode_reward_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)



    def setup_actor(self):
        import pdb; pdb.set_trace()
        model: PPOModel = instantiate(self.config.model)
        model.apply(weight_init)
        actor_optimizer = instantiate(
            self.config.model.config.actor_optimizer,
            params=list(model._actor.parameters()),
        )

        self.model, self.actor_optimizer = self.fabric.setup(
            model, actor_optimizer
        )
        self.model.mark_forward_method("act")
        self.model.mark_forward_method("get_action_and_value") # this might need to be changed
    
    def setup_expert(self, tag):
        import pdb; pdb.set_trace()
        expert: PPOModel = instantiate(self.config.expert)
        expert.apply(weight_init)
        expert.eval()
        self.experts[tag] = self.fabric.setup(expert)
        self.experts[tag].mark_forward_method("act")
        self.experts[tag].mark_forward_method("get_action_and_value") # this might need to be changed

    def load_actor(self, checkpoint: Path):
        if checkpoint is not None:
            checkpoint = Path(checkpoint).resolve()
            print(f"Loading model from checkpoint as actor: {checkpoint}")
            state_dict = torch.load(checkpoint, map_location=self.device)
            self.load_parameters(state_dict)
            
            env_checkpoint = checkpoint.resolve().parent / f"env_{self.fabric.global_rank}.ckpt"
            if env_checkpoint.exists():
                print(f"Loading env checkpoint: {env_checkpoint}")
                env_state_dict = torch.load(env_checkpoint, map_location=self.device)
                self.env.load_state_dict(env_state_dict)

    def load_actor_parameters(self, state_dict):
        self.current_epoch = state_dict["epoch"]

        if "step_count" in state_dict:
            print("step_count in state dict")
            self.step_count = state_dict["step_count"]
        if "run_start_time" in state_dict:
            self.fit_start_time = state_dict["run_start_time"]

        self.best_evaluated_score = state_dict.get("best_evaluated_score", None)

        self.model.load_state_dict(state_dict["model"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])

        self.episode_reward_meter.load_state_dict(state_dict["episode_reward_meter"])
        self.episode_length_meter.load_state_dict(state_dict["episode_length_meter"])
    
    def load_expert(self, checkpoint: Path, tag):
        if checkpoint is not None:
            checkpoint = Path(checkpoint).resolve()
            print(f"Loading model from checkpoint as expert for {tag}: {checkpoint}")
            state_dict = torch.load(checkpoint, map_location=self.device)
            self.load_expert_parameters(state_dict, tag)

    def load_expert_parameters(self, state_dict, tag):
        # expert_model_actor_only = self.remove_from_state_dict(state_dict["model"], '_critic.')
        import pdb; pdb.set_trace()
        expert_state_dict = self.remove_from_state_dict(state_dict["model"], "_critic.")
        self.experts[tag].load_state_dict(expert_state_dict)
    
    def remove_from_state_dict(self, state_dict, key_to_remove):
        keys = list(state_dict.keys())
        for key in keys:
            if key.startswith(key_to_remove):
                del state_dict[key]
        # return state_dict
        return state_dict

#     def run_training_loop(self, n_iter, collect_policy, eval_policy,
#                         initial_expertdata=None, relabel_with_expert=False,
#                         start_relabel_with_expert=1, expert_policy=None):
#         """
#         :param n_iter:  number of (dagger) iterations
#         :param collect_policy:
#         :param eval_policy:
#         :param initial_expertdata:
#         :param relabel_with_expert:  whether to perform dagger
#         :param start_relabel_with_expert: iteration at which to start relabel with expert
#         :param expert_policy:
#         """

#         # Initialize variables at beginning of training
#         self.total_envsteps = 0
#         self.start_time = time.time()

#         for itr in range(n_iter):
#             print("\n\n********** Iteration %i ************"%itr)

#             # # Decide if videos should be rendered/logged at this iteration
#             # if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
#             #     self.log_video = True
#             # else:
#             #     self.log_video = False

#             # # Decide if metrics should be logged
#             # if itr % self.params['scalar_log_freq'] == 0:
#             #     self.log_metrics = True
#             # else:
#             #     self.log_metrics = False

#             # Collect trajectories, to be used for training
#             training_returns = self.collect_training_trajectories(
#                 itr,
#                 initial_expertdata,
#                 collect_policy
#             )  # HW1: implement this function below
#             paths, envsteps_this_batch, train_video_paths = training_returns
#             self.total_envsteps += envsteps_this_batch

#             # Relabel the collected observations with actions from a provided expert policy
#             if relabel_with_expert and itr>=start_relabel_with_expert:
#                 # HW1: implement this function below
#                 paths = self.do_relabel_with_expert(expert_policy, paths)

#             # Add collected data to replay buffer
#             self.agent.add_to_replay_buffer(paths)

#             # Train agent (using sampled data from replay buffer)
#             # HW1: implement this function below
#             training_logs = self.train_agent()

#             # Log and save videos and metrics
#             # if self.log_video or self.log_metrics:

#             #     # Perform logging
#             #     print('\nBeginning logging procedure...')
#             #     self.perform_logging(
#             #         itr, paths, eval_policy, train_video_paths, training_logs)

#             #     if self.params['save_params']:
#             #         print('\nSaving agent params')
#             #         self.agent.save('{}/policy_itr_{}.pt'.format(self.params['logdir'], itr))


#     def collect_training_trajectories(
#             self,
#             itr,
#             load_initial_expertdata,
#             collect_policy
#     ):
#         """
#         :param itr:
#         :param load_initial_expertdata: path to expert data pkl file
#         :param collect_policy: the current policy using which we collect data
#         :return:
#             paths: a list trajectories
#             envsteps_this_batch: the sum over the numbers of environment steps in paths
#             train_video_paths: paths which also contain videos for visualization purposes
#         """

#         # TODO decide whether to load training data or use the current policy to collect more data
#         # HINT1: On the first iteration, do you need to collect training trajectories? You might
#         # want to handle loading from expert data, and if the data doesn't exist, collect an appropriate
#         # number of transitions.
#         # HINT2: Loading from expert transitions can be done using pickle.load()
#         # HINT3: To collect data, you might want to use pre-existing sample_trajectories code from utils
#         # HINT4: You want each of these collected rollouts to be of length self.params['ep_len']
#         if itr == 0 and load_initial_expertdata:
#             with open(load_initial_expertdata, 'rb') as file:
#                 paths = pickle.load(file)
#                 envsteps_this_batch = sum([utils.get_pathlength(path) for path in paths])
#         else:
#             print("\nCollecting data to be used for training...")
#             paths, envsteps_this_batch =  utils.sample_trajectories(self.params['eval_batch_size'])

    
# # def sample_trajectories(self, min_timesteps_per_batch):
# #     """
# #         Collect rollouts until we have collected `min_timesteps_per_batch` steps.
# #     """

# #     for step in range(min_timesteps_per_batch):
# #         import pdb; pdb.set_trace()
# #         obs = self.handle_reset(done_indices)
# #         self.experience_buffer.update_data("self_obs", step, obs["self_obs"])

# #         # TODO: ignore the value cause we dont have critic
# #         action, neglogp, _ = self.model.get_action_and_value(obs)
# #         self.experience_buffer.update_data("actions", step, action)
# #         self.experience_buffer.update_data("neglogp", step, neglogp)

# #         # Check for NaNs in observations and actions
# #         for key in obs.keys():
# #             if torch.isnan(obs[key]).any():
# #                 print(f"NaN in {key}: {obs[key]}")
# #                 raise ValueError("NaN in obs")
# #         if torch.isnan(action).any():
# #             raise ValueError(f"NaN in action: {action}")

# #         # Step the environment
# #         next_obs, rewards, dones, terminated, extras = self.env_step(action)

# #         all_done_indices = dones.nonzero(as_tuple=False)
# #         done_indices = all_done_indices.squeeze(-1)

# #         # # Update logging metrics with the environment feedback
# #         # self.post_train_env_step(rewards, dones, done_indices, extras, step)

# #         self.experience_buffer.update_data("rewards", step, rewards)
# #         self.experience_buffer.update_data("dones", step, dones)

# #         self.step_count += self.get_step_count_increment()
#     def sample_trajectories(self, policy, batch_size, ep_len):
#         paths = []
#         timesteps_collected = 0
#         done_indices = None

#         while timesteps_collected < batch_size:
#             path = {"obs": [], "acts": [], "dones": []}
#             obs = self.env.reset(done_indices)
#             for t in range(ep_len):
#                 action, _, _ = policy.get_action_and_value(obs)
#                 next_obs, rewards, dones, terminated, extras = self.env_step(action)
#                 path["obs"].append(obs["self_obs"])
#                 path["acts"].append(action)
#                 path["dones"].append(dones)
#                 obs = next_obs

#                 timesteps_collected += self.env.num_envs
#                 if timesteps_collected >= batch_size:
#                     break

#             paths.append({k: torch.stack(v) for k, v in path.items()})

#         return paths, timesteps_collected

#     # def handle_reset(self, done_indices):
#     #     pass
    
#     # def env_step(self, action):
#     #     pass
    
#     # def get_step_count_increment(self):
#     #     pass

#     def handle_reset(self, done_indices=None):
#         obs = self.env.reset(done_indices)
#         return obs

#     def env_step(self, actions):
#         obs, rewards, dones, extras = self.env.step(actions)
#         terminated = extras["terminate"]
#         return obs, rewards, dones, terminated, extras

#     def get_step_count_increment(self):
#         return self.env.num_envs * self.fabric.world_size