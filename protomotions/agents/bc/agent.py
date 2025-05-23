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
        self.num_mini_epochs: int = config.num_mini_epochs
        self.fit_start_time = None
        self.best_evaluated_score = None
        self.episode_reward_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)



    def setup_actor(self):
        # import pdb; pdb.set_trace()
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
        # import pdb; pdb.set_trace()
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
            self.load_actor_parameters(state_dict)
            
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
        expert_state_dict = self.remove_from_state_dict(state_dict["model"], "_critic.")
        self.experts[tag].load_state_dict(expert_state_dict)
    
    def remove_from_state_dict(self, state_dict, key_to_remove):
        keys = list(state_dict.keys())
        for key in keys:
            if key.startswith(key_to_remove):
                del state_dict[key]
        return state_dict
    
    def run_training_loop(self):
        self.experience_buffer = ExperienceBuffer(self.num_envs, self.num_steps).to(
                self.device
            )
        self.experience_buffer.register_key(
            "self_obs", shape=(self.env.config.robot.self_obs_size,)
        )
        self.experience_buffer.register_key(
            "actions", shape=(self.env.config.robot.number_of_actions,)
        )
        self.experience_buffer.register_key("rewards")
        self.experience_buffer.register_key("dones", dtype=torch.long)
        self.experience_buffer.register_key("neglogp")

        if self.config.get("extra_inputs", None) is not None:
            obs = self.env.get_obs()
            for key in self.config.extra_inputs.keys():
                assert (
                    key in obs
                ), f"Key {key} not found in obs returned from env: {obs.keys()}"
                env_tensor = obs[key]
                shape = env_tensor.shape
                dtype = env_tensor.dtype
                self.experience_buffer.register_key(key, shape=shape[1:], dtype=dtype)

        initial_expertdata = None

        while self.current_epoch < self.config.max_epochs:
            with torch.no_grad():
                self.fabric.call("before_play_steps", self)
                self.model.eval()
                if self.current_epoch >= 1:
                    relabel_with_expert = True
                else:
                    relabel_with_expert = False

                self.collect_training_trajectories(
                        self.current_epoch, # this may need to be iterations instead of step
                        initial_expertdata,
                        relabel_with_expert
                    )

            # training_log_dict["epoch"] = self.current_epoch
            # dataset = self.experience_buffer.make_dict()

            self.optimize_model()
            print("yay")

            self.current_epoch += 1
            self.fabric.call("after_train", self)

    def collect_training_trajectories(
            self,
            itr,
            load_initial_expertdata,
            use_expert
    ):
        """
        :param itr:
        :param load_initial_expertdata: path to expert data pkl file
        :param collect_policy: the current policy using which we collect data
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """

        # TODO decide whether to load training data or use the current policy to collect more data
        # HINT1: On the first iteration, do you need to collect training trajectories? You might
        # want to handle loading from expert data, and if the data doesn't exist, collect an appropriate
        # number of transitions.
        # HINT2: Loading from expert transitions can be done using pickle.load()
        # HINT3: To collect data, you might want to use pre-existing sample_trajectories code from utils
        # HINT4: You want each of these collected rollouts to be of length self.params['ep_len']
        # if itr == 0 and load_initial_expertdata:
        #     with open(load_initial_expertdata, 'rb') as file:
        #         paths = pickle.load(file)
        #         envsteps_this_batch = sum([utils.get_pathlength(path) for path in paths])
        # else:
        print("\nCollecting data to be used for training...")
        # eval_batch_size = 2
        motion_tag = list(self.experts.keys())[0]
        print(f"\nUsing {motion_tag} expert")
        self.sample_trajectories(self.num_steps, motion_tag, use_expert)

    
    def sample_trajectories(self, min_timesteps_per_batch, motion_tag, use_expert):
        """
            Collect rollouts until we have collected `min_timesteps_per_batch` steps.
        """
        done_indices = None
        for step in range(min_timesteps_per_batch):
            obs = self.handle_reset(done_indices)
            self.experience_buffer.update_data("self_obs", step, obs["self_obs"])
            if self.config.get("extra_inputs", None) is not None:
                for key in self.config.extra_inputs:
                    self.experience_buffer.update_data(key, step, obs[key])

            action, neglogp, _ = self.model.get_action_and_value(obs)
            if use_expert:
                expert_action, _, _ = self.experts[motion_tag].get_action_and_value(obs)
                self.experience_buffer.update_data("actions", step, expert_action)
            else:
                self.experience_buffer.update_data("actions", step, action)
            self.experience_buffer.update_data("neglogp", step, neglogp)

            # Check for NaNs in observations and actions
            for key in obs.keys():
                if torch.isnan(obs[key]).any():
                    print(f"NaN in {key}: {obs[key]}")
                    raise ValueError("NaN in obs")
            if torch.isnan(action).any():
                raise ValueError(f"NaN in action: {action}")

            # Step the environment
            next_obs, rewards, dones, terminated, extras = self.env_step(action)

            all_done_indices = dones.nonzero(as_tuple=False)
            done_indices = all_done_indices.squeeze(-1)

            # # Update logging metrics with the environment feedback
            # self.post_train_env_step(rewards, dones, done_indices, extras, step)

            self.experience_buffer.update_data("rewards", step, rewards)
            self.experience_buffer.update_data("dones", step, dones)

            self.step_count += self.get_step_count_increment()


    def handle_reset(self, done_indices=None):
        obs = self.env.reset(done_indices)
        return obs
    

    def env_step(self, actions):
        obs, rewards, dones, extras = self.env.step(actions)
        terminated = extras["terminate"]
        return obs, rewards, dones, terminated, extras


    def get_step_count_increment(self):
        return self.env.num_envs * self.fabric.world_size


    # -----------------------------
    # Optimization
    # -----------------------------
    def optimize_model(self) -> Dict:
        # training/optimization loop

        dataset = self.process_dataset(self.experience_buffer.make_dict())
        self.train()
        training_log_dict = {}

        for batch_idx in track(
            range(self.max_num_batches()),
            description=f"Epoch {self.current_epoch}, training...",
        ):
            iter_log_dict = {}
            dataset_idx = batch_idx % len(dataset)
            if dataset_idx == 0 and batch_idx != 0 and dataset.do_shuffle:
                dataset.shuffle()            
            batch_dict = dataset[dataset_idx]

            # Go over all keys in obs and check if any has nans
            for key in batch_dict.keys():
                has_nan = False
                if torch.isnan(batch_dict[key]).any():
                    has_nan = True
                    print(f"NaN in {key}: {batch_dict[key]}")
                if has_nan:
                    raise ValueError("NaN in training")
   
            loss_dict = self.model_step(batch_dict)
            iter_log_dict.update(loss_dict)

            for k, v in iter_log_dict.items():
                if k in training_log_dict:
                    training_log_dict[k][0] += v
                    training_log_dict[k][1] += 1
                else:
                    training_log_dict[k] = [v, 1]

        for k, v in training_log_dict.items():
            training_log_dict[k] = v[0] / v[1]

        self.eval()
        return training_log_dict
        
        


        # for batch_idx in track(
        #     range(self.max_num_batches()),
        #     description=f"Epoch {self.current_epoch}, training...",
        # ):
        #     iter_log_dict = {}
        #     dataset_idx = batch_idx % len(dataset)

        #     # Reshuffle dataset at the beginning of each mini epoch if configured.
        #     if dataset_idx == 0 and batch_idx != 0 and dataset.do_shuffle:
        #         dataset.shuffle()
        #     batch_dict = dataset[dataset_idx]

        #     # Check for NaNs in the batch.
        #     for key in batch_dict.keys():
        #         if torch.isnan(batch_dict[key]).any():
        #             print(f"NaN in {key}: {batch_dict[key]}")
        #             raise ValueError("NaN in training")

        #     # Update actor
        #     actor_loss, actor_loss_dict = self.actor_step(batch_dict)
        #     iter_log_dict.update(actor_loss_dict)
        #     self.actor_optimizer.zero_grad(set_to_none=True)
        #     self.fabric.backward(actor_loss)
        #     actor_grad_clip_dict = self.handle_model_grad_clipping(
        #         self.model._actor, self.actor_optimizer, "actor"
        #     )
        #     iter_log_dict.update(actor_grad_clip_dict)
        #     self.actor_optimizer.step()

        #     # Update critic
        #     critic_loss, critic_loss_dict = self.critic_step(batch_dict)
        #     iter_log_dict.update(critic_loss_dict)
        #     self.critic_optimizer.zero_grad(set_to_none=True)
        #     self.fabric.backward(critic_loss)
        #     critic_grad_clip_dict = self.handle_model_grad_clipping(
        #         self.model._critic, self.critic_optimizer, "critic"
        #     )
        #     iter_log_dict.update(critic_grad_clip_dict)
        #     self.critic_optimizer.step()

        #     # Extra optimization steps if needed.
        #     extra_opt_steps_dict = self.extra_optimization_steps(batch_dict, batch_idx)
        #     iter_log_dict.update(extra_opt_steps_dict)

        #     for k, v in iter_log_dict.items():
        #         if k in training_log_dict:
        #             training_log_dict[k][0] += v
        #             training_log_dict[k][1] += 1
        #         else:
        #             training_log_dict[k] = [v, 1]

        # for k, v in training_log_dict.items():
        #     training_log_dict[k] = v[0] / v[1]

        # self.eval()
        # return training_log_dict

    # -----------------------------
    # Helper Functions
    # -----------------------------
    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()
    
    def model_step(self, batch_dict):
        """
        Update the policy using expert demonstrations
        """
        # Get policy's action distribution
        dist = self.model._actor(batch_dict)
        # expert_actions = batch_dict["actions"]
        # loss = torch.square(dist.mean - expert_actions).mean()

        logstd = self.model._actor.logstd
        std = torch.exp(logstd)
        neglogp = self.model.neglogp(batch_dict["actions"], dist.mean, std, logstd)
        loss = neglogp.mean()    
        
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.fabric.backward(loss)
        self.actor_optimizer.step()
        
        return {"model/loss": loss.detach().item()}

    @torch.no_grad()
    def process_dataset(self, dataset):
        # if self.config.normalize_values:
        #     self.running_val_norm.update(dataset["values"])
        #     self.running_val_norm.update(dataset["returns"])

        #     dataset["values"] = self.running_val_norm.normalize(dataset["values"])
        #     dataset["returns"] = self.running_val_norm.normalize(dataset["returns"])

        dataset = DictDataset(self.config.batch_size, dataset, shuffle=True)
        return dataset

    def max_num_batches(self):
        return math.ceil(
            self.num_envs * self.num_steps * self.num_mini_epochs / self.config.batch_size
        )
