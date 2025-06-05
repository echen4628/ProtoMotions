import numpy as np


def handle_reset(env, force_full_restart=False, done_indices=None):
    if force_full_restart:
        done_indices = None
    obs = env.reset(done_indices)
    return obs

def env_step(env, actions):
    obs, rewards, dones, extras = env.step(actions)
    # rewards = rewards * task_reward_w
    terminated = extras["terminate"]
    return obs, rewards, dones, terminated, extras

def sample_trajectory(env, policy, max_path_length, render=False):
    """
    Rolls out a policy and generates a trajectories

    :param policy: the policy to roll out
    :param max_path_length: the number of steps to roll out
    :render: whether to save images from the rollout
    """
    policy.eval()
    done_indices = None  # Force reset on first entry
    step = 0
    while step < max_path_length:
        obs = handle_reset(env, done_indices=done_indices)
        # Obtain actor predictions
        # needs to be done in parallel
        actions = policy.act(obs)
        # Step the environment
        obs, rewards, dones, terminated, extras = env_step(env, actions)
        all_done_indices = dones.nonzero(as_tuple=False)
        done_indices = all_done_indices.squeeze(-1)
        step += 1
    
    
    # # Initialize environment for the beginning of a new rollout
    # ob = env.reset() # HINT: should be the output of resetting the env
    # # Initialize data storage for across the trajectory
    # # You'll mainly be concerned with: obs (list of observations), acs (list of actions)
    # obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    # steps = 0
    # while True:

    #     # Render image of the simulated environment
    #     if render:
    #         if hasattr(env.unwrapped, 'sim'):
    #             if 'track' in env.unwrapped.model.camera_names:
    #                 image_obs.append(env.unwrapped.sim.render(camera_name='track', height=500, width=500)[::-1])
    #             else:
    #                 image_obs.append(env.unwrapped.sim.render(height=500, width=500)[::-1])
    #         else:
    #             image_obs.append(env.render())

    #     # Use the most recent observation to decide what to do
    #     obs.append(ob)
    #     ac = policy.get_action(np.array(ob)) # HINT: Query the policy's get_action function
    #     ac = ac[0]
    #     acs.append(ac)

    #     # Take that action and record results
    #     ob, rew, done, _ = env.step(ac)

    #     # Record result of taking that action
    #     steps += 1
    #     next_obs.append(ob)
    #     rewards.append(rew)

    #     # TODO end the rollout if the rollout ended
    #     # HINT: rollout can end due to done, or due to max_path_length
    #     rollout_done = 1 if (done or steps==max_path_length) else 0 # HINT: this is either 0 or 1
    #     terminals.append(rollout_done)

    #     if rollout_done:
    #         break

    # return Path(obs, image_obs, acs, rewards, next_obs, terminals)

def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False):
    """
        Collect rollouts until we have collected `min_timesteps_per_batch` steps.
    """
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        paths.append(sample_trajectory(env, policy, max_path_length, render))
        timesteps_this_batch += get_pathlength(paths[-1])

    return paths, timesteps_this_batch

    for step in range(min_timesteps_per_batch):
        import pdb; pdb.set_trace()
        obs = self.handle_reset(done_indices)
        self.experience_buffer.update_data("self_obs", step, obs["self_obs"])

        # TODO: ignore the value cause we dont have critic
        action, neglogp, _ = self.model.get_action_and_value(obs)
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
