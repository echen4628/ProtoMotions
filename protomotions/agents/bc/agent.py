import torch
import os
import logging

from torch import Tensor

import time
import math
from pathlib import Path
from typing import Optional, Tuple, Dict
import pickle

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

        self.action_dim = self.env.config.robot.number_of_actions
        self.num_envs = self.env.config.num_envs


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
        initial_expertdata = None  # this is always None; remove it!

        while self.current_epoch < self.config.max_epochs:
            with torch.no_grad():
                self.fabric.call("before_play_steps", self)
                self.model.eval()

                self.sample_trajectories(self.num_steps, use_expert=True)

                # recommended to clear the experience buffer after each epoch?
                # self.collect_training_trajectories(
                #         self.current_epoch, # this may need to be iterations instead of step
                #         initial_expertdata,
                #         use_expert=True
                #     )

            # training_log_dict["epoch"] = self.current_epoch
            # dataset = self.experience_buffer.make_dict()

            # # training/optimization loop
            # for batch_idx in track(
            #     range(self.max_num_batches()),
            #     description=f"Epoch {self.current_epoch}, training...",
            # ):
            #     dataset_idx = batch_idx % len(dataset)
            #     if dataset_idx == 0 and batch_idx != 0 and dataset.do_shuffle:
            #         dataset.shuffle()            
            #     batch_dict = dataset[dataset_idx]

            #     # Go over all keys in obs and check if any has nans
            #     for key in batch_dict.keys():
            #         has_nan = False
            #         if torch.isnan(batch_dict[key]).any():
            #             has_nan = True
            #             print(f"NaN in {key}: {batch_dict[key]}")
            #         if has_nan:
            #             raise ValueError("NaN in training")
                    
            #     self.model_step(batch_dict)

            self.current_epoch += 1
            self.fabric.call("after_train", self)

    # def collect_training_trajectories(
    #         self,
    #         itr,
    #         load_initial_expertdata,
    #         use_expert
    # ):
        """
        :param itr:
        :param load_initial_expertdata: path to expert data pkl file
        :param collect_policy: the current policy using which we collect data
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """

        """
        assuming self.experts -> {
                        "0": {
                            "label": "",
                            "ckpt_path": ""
                        }, 
                        ....}
        """

        # for _, metadata in self.experts.items():
        #     motion_tag = metadata["label"]
        #     self.sample_trajectories(self.num_steps, motion_tag, use_expert)


        


        


        # self.env.motion_lib.motion_ids, self.env.motion_lib.motion_files

        # for motion_tag in self.experts.keys():
        #     print(f"\nUsing {motion_tag} expert")
        #     # Collect data for this expert; you may want to split self.num_steps among experts
        #     # For example, use self.num_steps // len(self.experts) to distribute steps evenly
        #     expert_steps = self.num_steps // len(self.experts)
        #     # Collect trajectories for this expert
        #     self.sample_trajectories(expert_steps, motion_tag, use_expert)
            # paths.extend(expert_paths)
            




        # print("\nCollecting data to be used for training...")
        # eval_batch_size = 2
        # motion_tag = list(self.experts.keys())[0]
        # print(f"\nUsing {motion_tag} expert")
        # self.sample_trajectories(self.num_steps, motion_tag, use_expert)

    
    def sample_trajectories(self, min_timesteps_per_batch, use_expert):
        """
            Collect rollouts until we have collected `min_timesteps_per_batch` steps.
        """
        done_indices = None
        for step in range(min_timesteps_per_batch):
            obs = self.handle_reset(done_indices)
            self.experience_buffer.update_data("self_obs", step, obs["self_obs"])

            action_experts = torch.zeros((self.num_envs, self.action_dim))  # Adjust output_dim accordingly
            # action_model = torch.zeros((self.num_envs, self.action_dim)) # TODO move to GPU?
            motion_ids = obs["motion_ids"]  # [4096] -> 0, 1, 2,.. N-1 (N = #motions)
            for motion_id, model in enumerate(self.experts.items()):  # models = [model0, model1, model2]
                mask = (motion_ids == motion_id)
                if mask.any():
                    partial_obs = obs[mask]
                    expert_action, _, _ = model.get_action_and_value(partial_obs)
                    action_experts[mask] = expert_action

            action_model, neglogp, _ = self.model.get_action_and_value(partial_obs)
            if use_expert:
                self.experience_buffer.update_data("actions", step, action_experts)
            else:
                self.experience_buffer.update_data("actions", step, action_model)
            self.experience_buffer.update_data("neglogp", step, neglogp)


            # import pdb; pdb.set_trace()
            # # TODO: ignore the value cause we dont have critic
            # action, neglogp, _ = self.model.get_action_and_value(obs)
            # if use_expert:
            #     expert_action, _, _ = self.experts[motion_tag].get_action_and_value(obs)
            #     self.experience_buffer.update_data("actions", step, expert_action)
            # else:
            #     self.experience_buffer.update_data("actions", step, action)
            # self.experience_buffer.update_data("neglogp", step, neglogp)

            # Check for NaNs in observations and actions
            for key in obs.keys():
                if torch.isnan(obs[key]).any():
                    print(f"NaN in {key}: {obs[key]}")
                    raise ValueError("NaN in obs")
            if torch.isnan(action_model).any():
                raise ValueError(f"NaN in action: {action_model}")

            # Step the environment
            next_obs, rewards, dones, terminated, extras = self.env_step(action_model)

            all_done_indices = dones.nonzero(as_tuple=False)
            done_indices = all_done_indices.squeeze(-1)

            # # Update logging metrics with the environment feedback
            # self.post_train_env_step(rewards, dones, done_indices, extras, step)

            self.experience_buffer.update_data("rewards", step, rewards)
            self.experience_buffer.update_data("dones", step, dones)

            self.step_count += self.get_step_count_increment()


    def handle_reset(self, done_indices=None):
        import pdb; pdb.set_trace()
        obs = self.env.reset(done_indices)
        return obs
    

    def env_step(self, actions):
        obs, rewards, dones, extras = self.env.step(actions)
        terminated = extras["terminate"]
        return obs, rewards, dones, terminated, extras


    def get_step_count_increment(self):
        return self.env.num_envs * self.fabric.world_size
    
    def model_step(self, batch_dict):
        """
        Update the policy using expert demonstrations
        """
        # Get policy's action distribution
        dist = self.model._actor(batch_dict)
        expert_actions = batch_dict["expert_actions"]
        bc_loss = torch.square(dist.mean - expert_actions).mean()

        # logstd = self.model._actor.logstd
        # std = torch.exp(logstd)
        # neglogp = self.model.neglogp(batch_dict["actions"], dist.mean, std, logstd)
        # loss = neglogp.mean()    
        
        self.optimizer.zero_grad()
        self.fabric.backward(bc_loss)
        self.optimizer.step()
        
        return {"model/bc_loss": bc_loss.detach().item()}