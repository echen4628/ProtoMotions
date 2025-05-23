import torch
import os
import logging

from torch import Tensor

import time
import math
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from lightning.fabric import Fabric

from hydra.utils import instantiate
from isaac_utils import torch_utils

from protomotions.utils.time_report import TimeReport
from protomotions.utils.average_meter import AverageMeter, TensorAverageMeterDict
from protomotions.agents.utils.data_utils import DictDataset, ExperienceBuffer
from protomotions.agents.ppo.model import PPOModel
from protomotions.agents.common.common import weight_init, get_params
from protomotions.envs.base_env.env import BaseEnv
from protomotions.envs.mimic.env import Mimic as MimicEnv
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
    
    def get_state_dict(self, state_dict):
        extra_state_dict = {
            "model": self.model.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "epoch": self.current_epoch,
            "step_count": self.step_count,
            "run_start_time": self.fit_start_time,
            "episode_reward_meter": self.episode_reward_meter.state_dict(),
            "episode_length_meter": self.episode_length_meter.state_dict(),
            "best_evaluated_score": self.best_evaluated_score,
        }

        state_dict.update(extra_state_dict)
        return state_dict

    def save(self, path=None, name="last.ckpt", new_high_score=False):
        if path is None:
            path = self.fabric.loggers[0].log_dir
        root_dir = Path.cwd() / Path(self.fabric.loggers[0].root_dir)
        save_dir = Path.cwd() / Path(path)
        state_dict = self.get_state_dict({})
        self.fabric.save(save_dir / name, state_dict)

        if self.fabric.global_rank == 0:
            if root_dir != save_dir:
                if (root_dir / "last.ckpt").is_symlink():
                    (root_dir / "last.ckpt").unlink()
                # Make root_dir / "last.ckpt" point to the new checkpoint.
                # Calculate the relative path and create a symbolic link.
                relative_path = Path(os.path.relpath(save_dir / name, root_dir))
                (root_dir / "last.ckpt").symlink_to(relative_path)
                log.info(f"saved checkpoint, {root_dir / 'last.ckpt'}")
        self.fabric.barrier()
        
        # Save env state for all ranks to the same directory.
        rank_0_path = (root_dir / "last.ckpt").resolve().parent
        env_checkpoint = rank_0_path / f"env_{self.fabric.global_rank}.ckpt"
        env_state_dict = self.env.get_state_dict()
        torch.save(env_state_dict, env_checkpoint)

        # Check if new high score flag is consistent across devices.
        gathered_high_score = self.fabric.all_gather(new_high_score)
        assert all(
            [x == gathered_high_score[0] for x in gathered_high_score]
        ), "New high score flag should be the same across all ranks."

        if new_high_score:
            score_based_name = "score_based.ckpt"
            self.fabric.save(save_dir / score_based_name, state_dict)
            print(
                f"New best performing controller found with score {self.best_evaluated_score}. Model saved to {save_dir / score_based_name}."
            )
            if self.fabric.global_rank == 0:
                if root_dir != save_dir:
                    if (root_dir / "score_based.ckpt").is_symlink():
                        (root_dir / "score_based.ckpt").unlink()
                    # Create symlink for the best score checkpoint.
                    relative_path = Path(os.path.relpath(save_dir / name, root_dir))
                    (root_dir / "score_based.ckpt").symlink_to(relative_path)

    
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
        initial_expertdata = None

        if self.fit_start_time is None:
            self.fit_start_time = time.time()
        while self.current_epoch < self.config.max_epochs:
            self.epoch_start_time = time.time()
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
            # TODO: add other necessary values into training_log_dict
            training_log_dict = {}
            training_log_dict["epoch"] = self.current_epoch

            dataset = self.experience_buffer.make_dict()

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


            # Save model checkpoint at specified intervals before evaluation.
            if self.current_epoch % self.config.manual_save_every == 0:
                self.save()
            if (
                self.config.eval_metrics_every is not None
                and self.current_epoch > 0
                and self.current_epoch % self.config.eval_metrics_every == 0
            ):
                eval_log_dict, evaluated_score = self.calc_eval_metrics()
                evaluated_score = self.fabric.broadcast(evaluated_score, src=0)
                if evaluated_score is not None:
                    if (
                        self.best_evaluated_score is None
                        or evaluated_score >= self.best_evaluated_score
                    ):
                        self.best_evaluated_score = evaluated_score
                        self.save(new_high_score=True)
                training_log_dict.update(eval_log_dict) 

            self.post_epoch_logging(training_log_dict)
            self.env.on_epoch_end(self.current_epoch)

        self.save()

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
        eval_batch_size = 2
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

            # TODO: ignore the value cause we dont have critic
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
    

    # @torch.no_grad()
    # def calc_eval_metrics(self) -> Tuple[Dict, Optional[float]]:
    #     return {}, None

    @torch.no_grad()
    def evaluate_policy(self):
        self.eval()
        done_indices = None  # Force reset on first entry
        step = 0
        while self.config.max_eval_steps is None or step < self.config.max_eval_steps:
            obs = self.handle_reset(done_indices)
            # Obtain actor predictions
            actions = self.model.act(obs)
            # Step the environment
            obs, rewards, dones, terminated, extras = self.env_step(actions)
            print(rewards)
            all_done_indices = dones.nonzero(as_tuple=False)
            done_indices = all_done_indices.squeeze(-1)
            step += 1
    
    def post_epoch_logging(self, training_log_dict: Dict):
        end_time = time.time()
        log_dict = {
            "info/episode_length": self.episode_length_meter.get_mean().item(),
            "info/episode_reward": self.episode_reward_meter.get_mean().item(),
            "info/frames": torch.tensor(self.step_count),
            "info/gframes": torch.tensor(self.step_count / (10**9)),
            "times/fps_last_epoch": (self.num_steps * self.get_step_count_increment())
            / (end_time - self.epoch_start_time),
            "times/fps_total": self.step_count / (end_time - self.fit_start_time),
            "times/training_hours": (end_time - self.fit_start_time) / 3600,
            "times/training_minutes": (end_time - self.fit_start_time) / 60,
            "times/last_epoch_seconds": (end_time - self.epoch_start_time),
            # "rewards/task_rewards": self.experience_buffer.rewards.mean().item(),
            # "rewards/extra_rewards": self.experience_buffer.extra_rewards.mean().item(),
            # "rewards/total_rewards": self.experience_buffer.total_rewards.mean().item(),
        }
        # env_log_dict = self.episode_env_tensors.mean_and_clear()
        # env_log_dict = {f"env/{k}": v for k, v in env_log_dict.items()}
        # if len(env_log_dict) > 0:
        #     log_dict.update(env_log_dict)
        log_dict.update(training_log_dict)
        self.fabric.log_dict(log_dict)
    
    def eval(self):
        self.model.eval()

    def map_motions_to_iterations(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Maps motion IDs to iterations for distributed processing.

        This method distributes motion IDs across available ranks and creates a mapping
        of motions to be processed in each iteration. It ensures equal distribution of
        motions across ranks and proper scene sampling.

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: A list of tuples, each containing:
                - motion_ids: Tensor of motion IDs for an iteration.
                - requires_scene: Tensor of booleans indicating if the motion requires a scene.
        """
        world_size = self.fabric.world_size
        global_rank = self.fabric.global_rank
        num_motions = self.motion_lib.num_motions()

        # Handle fixed motion ID case
        if self.env.config.motion_manager.fixed_motion_id is not None:
            motion_ids = torch.tensor(
                [self.env.config.motion_manager.fixed_motion_id], device=self.device
            )
            requires_scene = self.env.get_motion_requires_scene(motion_ids)
            return [(motion_ids, requires_scene)], 1

        # Calculate motions per rank, ensuring even distribution
        base_motions_per_rank = num_motions // world_size
        extra_motions = num_motions % world_size

        # Ranks with index < extra_motions get one additional motion
        motions_per_rank = base_motions_per_rank + (1 if global_rank < extra_motions else 0)
        start_motion = base_motions_per_rank * global_rank + min(global_rank, extra_motions)
        end_motion = start_motion + motions_per_rank

        # Create tensor of motion IDs assigned to this rank
        motion_range = torch.arange(start_motion, end_motion, device=self.device)

        # Split motions into batches of size self.num_envs
        motion_map = []
        for i in range(0, len(motion_range), self.num_envs):
            batch_motion_ids = motion_range[i : i + self.num_envs]
            # Sample corresponding scene IDs
            requires_scene = self.env.get_motion_requires_scene(batch_motion_ids)
            motion_map.append((batch_motion_ids, requires_scene))

        return motion_map, motions_per_rank

    @torch.no_grad()
    def calc_eval_metrics(self) -> Tuple[Dict, Optional[float]]:
        self.eval()
        if self.env.config.motion_manager.fixed_motion_id is not None:
            num_motions = 1
        else:
            num_motions = self.motion_lib.num_motions()

        metrics = {
            # Track which motions are evaluated (within time limit)
            "evaluated": torch.zeros(num_motions, device=self.device, dtype=torch.bool),
        }
        for k in self.config.eval_metric_keys:
            metrics[k] = torch.zeros(num_motions, device=self.device)
            metrics[f"{k}_max"] = torch.zeros(num_motions, device=self.device)
            metrics[f"{k}_min"] = 3 * torch.ones(num_motions, device=self.device)

        # Compute how many motions each rank should evaluate
        root_dir = Path(self.fabric.loggers[0].root_dir)
        motion_map, remaining_motions = self.map_motions_to_iterations()
        num_outer_iters = len(motion_map)
        # Maximal number of iterations any of the ranks needs to perform.
        max_iters = max(self.fabric.all_gather(len(motion_map)))

        for outer_iter in track(
            range(max_iters),
            description=f"Evaluating... {remaining_motions} motions remain...",
        ):
            motion_pointer = outer_iter % num_outer_iters
            motion_ids, requires_scene = motion_map[motion_pointer]
            num_motions_this_iter = len(motion_ids)
            metrics["evaluated"][motion_ids] = True

            # Define the task mapping for each agent.
            self.env.agent_in_scene[:] = False
            self.env.motion_manager.motion_ids[:num_motions_this_iter] = motion_ids
            self.env.agent_in_scene[:num_motions_this_iter] = requires_scene
            # Force respawn on flat terrain to ensure proper motion reconstruction.
            self.env.force_respawn_on_flat = True

            env_ids = torch.arange(
                0, num_motions_this_iter, dtype=torch.long, device=self.device
            )

            dt: float = self.env.dt
            motion_lengths = self.motion_lib.get_motion_length(motion_ids)
            motion_num_frames = (motion_lengths / dt).floor().long()

            max_len = (
                motion_num_frames.max().item()
                if self.config.eval_length is None
                else self.config.eval_length
            )

            for eval_episode in range(self.config.eval_num_episodes):
                # Sample random start time with slight noise for varied initial conditions.
                elapsed_time = (
                    torch.rand_like(self.motion_lib.state.motion_lengths[motion_ids]) * dt
                )
                self.env.motion_manager.motion_times[:num_motions_this_iter] = elapsed_time
                self.env.motion_manager.reset_track_steps.reset_steps(env_ids)
                # Disable automatic reset to maintain consistency in evaluation.
                self.env.disable_reset = True
                self.env.motion_manager.disable_reset_track = True

                obs = self.env.reset(
                    torch.arange(0, self.num_envs, dtype=torch.long, device=self.device)
                )

                for l in range(max_len):
                    actions = self.model.act(obs)
                    obs, rewards, dones, terminated, extras = self.env_step(actions)
                    elapsed_time += dt
                    clip_done = (motion_lengths - dt) < elapsed_time
                    clip_not_done = torch.logical_not(clip_done)
                    for k in self.config.eval_metric_keys:
                        if k in self.env.mimic_info_dict:
                            value = self.env.mimic_info_dict[k].detach()
                        else:
                            raise ValueError(f"Key {k} not found in mimic_info_dict")
                        # Only update metrics for motions that are continuing.
                        metric = value[:num_motions_this_iter]
                        metrics[k][motion_ids[clip_not_done]] += metric[clip_not_done]
                        metrics[f"{k}_max"][motion_ids[clip_not_done]] = torch.maximum(
                            metrics[f"{k}_max"][motion_ids[clip_not_done]],
                            metric[clip_not_done],
                        )
                        metrics[f"{k}_min"][motion_ids[clip_not_done]] = torch.minimum(
                            metrics[f"{k}_min"][motion_ids[clip_not_done]],
                            metric[clip_not_done],
                        )

        print("Evaluation done, now aggregating data.")

        if self.env.config.motion_manager.fixed_motion_id is None:
            motion_lengths = self.motion_lib.state.motion_lengths[:]
            motion_num_frames = (motion_lengths / dt).floor().long()

        # Save metrics per rank; distributed all_gather does not support dictionaries.
        with open(root_dir / f"{self.fabric.global_rank}_metrics.pt", "wb") as f:
            torch.save(metrics, f)
        self.fabric.barrier()
        # All ranks aggregrate data from all ranks.
        for rank in range(self.fabric.world_size):
            with open(root_dir / f"{rank}_metrics.pt", "rb") as f:
                other_metrics = torch.load(f, map_location=self.device)
            other_evaluated_indices = torch.nonzero(other_metrics["evaluated"]).flatten()
            for k in other_metrics.keys():
                metrics[k][other_evaluated_indices] = other_metrics[k][other_evaluated_indices]
            metrics["evaluated"][other_evaluated_indices] = True

        assert metrics["evaluated"].all(), "Not all motions were evaluated."
        self.fabric.barrier()
        (root_dir / f"{self.fabric.global_rank}_metrics.pt").unlink()

        to_log = {}
        for k in self.config.eval_metric_keys:
            mean_tracking_errors = metrics[k] / (motion_num_frames * self.config.eval_num_episodes)
            to_log[f"eval/{k}"] = mean_tracking_errors.detach().mean().item()
            to_log[f"eval/{k}_max"] = metrics[f"{k}_max"].detach().mean().item()
            to_log[f"eval/{k}_min"] = metrics[f"{k}_min"].detach().mean().item()

        if "gt_err" in self.config.eval_metric_keys:
            tracking_failures = (metrics["gt_err_max"] > 0.5).float()
            to_log["eval/tracking_success_rate"] = 1.0 - tracking_failures.detach().mean().item()

            failed_motions = torch.nonzero(tracking_failures).flatten().tolist()
            print(f"Saving to: {root_dir / f'failed_motions_{self.fabric.global_rank}.txt'}")
            with open(root_dir / f"failed_motions_{self.fabric.global_rank}.txt", "w") as f:
                for motion_id in failed_motions:
                    f.write(f"{motion_id}\n")
                    
            new_weights = torch.ones(self.motion_lib.num_motions(), device=self.device) * 1e-4
            new_weights[failed_motions] = 1.0
            self.env.motion_manager.update_sampling_weights(new_weights)

        stop_early = (
            self.config.training_early_termination.early_terminate_cart_err is not None
            or self.config.training_early_termination.early_terminate_success_rate is not None
        ) and self.fabric.global_rank == 0

        if self.config.training_early_termination.early_terminate_cart_err is not None:
            cart_err = to_log["eval/cartesian_err"]
            stop_early = stop_early and (cart_err <= self.config.training_early_termination.early_terminate_cart_err)
        if self.config.training_early_termination.early_terminate_success_rate is not None:
            tracking_success_rate = to_log["eval/tracking_success_rate"]
            stop_early = stop_early and (tracking_success_rate >= self.config.training_early_termination.early_terminate_success_rate)

        if stop_early:
            print("Stopping early! Target error reached")
            if "tracking_success_rate" in self.config.eval_metric_keys:
                print(f"tracking_success_rate: {to_log['eval/tracking_success_rate']}")
            if "cartesian_err" in self.config.eval_metric_keys:
                print(f"cartesian_err: {to_log['eval/cartesian_err']}")
            evaluated_score = self.fabric.broadcast(to_log["eval/tracking_success_rate"], src=0)
            self.best_evaluated_score = evaluated_score
            self.save(new_high_score=True)
            self.terminate_early()

        self.env.disable_reset = False
        self.env.motion_manager.disable_reset_track = False
        self.env.force_respawn_on_flat = False

        all_ids = torch.arange(0, self.num_envs, dtype=torch.long, device=self.device)
        self.env.motion_manager.reset_envs(all_ids)
        self.force_full_restart = True

        return to_log, to_log.get("eval/tracking_success_rate", to_log.get("eval/cartesian_err", None))
