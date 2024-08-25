# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from typing import Optional
from config import Args
from actor_critic import *
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from buffer import ReplayBuffer,PrioritizedReplayBuffer
from torch.utils.tensorboard import SummaryWriter

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.buffer_size = int(args.buffer_size)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if not args.evaluate:
        if args.track:
            import wandb
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                group=args.exp_name,
                save_code=True,
            )
            print(f"Logging to wandb in project {args.wandb_project_name} with name {run_name}")
        writer = SummaryWriter(f"runs/{run_name}")
        print(f"Logging to tensorboard in runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    else:
        writer = None
        eval_rewards = []
        print("Evaluation mode is activated, will not track the experiment")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_kwargs = dict(obs_mode="state", control_mode="pd_joint_delta_pos", render_mode="rgb_array", sim_backend="gpu")
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, **env_kwargs)    
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate and args.checkpoint!="":
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.evaluate, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, **env_kwargs)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.partial_reset, **env_kwargs)
    assert isinstance(envs.unwrapped.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.unwrapped.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())


    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.unwrapped.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha
    
    if args.checkpoint!="":
        actor,qf1,qf2,qf1_target,qf2_target,alpha = load_model(actor, qf1, qf2, qf1_target, qf2_target, alpha, args.checkpoint)
        print(f"Model loaded from {args.checkpoint}")
    
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # TODO: Change the buffer manamgement stratagy
    rb = PrioritizedReplayBuffer(
        args,
        args.buffer_size,
        envs.unwrapped.single_observation_space,
        envs.unwrapped.single_action_space,
        device,
        n_envs=args.num_envs
    )


    
    start_time = time.time()
    if args.evaluate:
        obs, _ = eval_envs.reset(seed=args.seed)
    else:
        obs, _ = envs.reset(seed=args.seed)

    episodic_return = torch.zeros(args.num_envs).to(device)
    episodic_length = torch.zeros(args.num_envs).to(device)
    done_count = 0
    avg_return = 0
    avg_length = 0
    success_count = 0
    for global_step in range(args.total_timesteps):

        env_step = global_step*args.num_envs
        
        # Training : collect data and store in the replay buffer
        if not args.evaluate:
            if global_step < args.learning_starts:
                actions = torch.tensor(envs.action_space.sample(), device=device)
            else:
                assert obs.shape[0] == args.num_envs, "The observation is not batched"
                actions, _, _ = actor.get_action(obs)

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            dones = truncations | terminations
            episodic_return += rewards
            episodic_length += 1


            if dones.any():
                avg_return += torch.sum(episodic_return[dones]).item()
                avg_length += torch.sum(episodic_length[dones]).item()
                success_count += torch.sum(infos["success"]).item()
                done_count += torch.sum(dones).item()
                if args.RESCALE_REWARDS:
                    rewards[infos["success"]]*=100
                episodic_return[dones] = 0
                episodic_length[dones] = 0

            rb.add(obs.cpu().detach().numpy(), next_obs.cpu().detach().numpy(), actions.cpu().detach().numpy(), rewards.cpu().detach().numpy(), terminations.cpu().detach().numpy(), infos)

        # Evaluation
        else:
            actions, _, _ = actor.get_action(obs)

            next_obs, rewards, terminations, truncations, _ = eval_envs.step(actions.cpu().detach().numpy())
            dones = terminations|truncations

            if args.num_eval_envs == 1:
                eval_rewards.append(rewards.item())
            else:
                eval_rewards.append(rewards.mean().item())
            
            if dones.any():
                print(f"env_step={env_step}, episodic_return={sum(eval_rewards)}")
                eval_rewards = []
                next_obs, _ = eval_envs.reset(seed=args.seed)


        obs = next_obs.clone()

        # Training : update the network
        if global_step > args.learning_starts and not args.evaluate:
            data, sample_batch_inds, sample_env_indices, sample_weights = rb.sample(args.batch_size)
            sample_weights = torch.FloatTensor(sample_weights).to(device).unsqueeze(1)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_error = next_q_value - qf1_a_values
            qf2_error = next_q_value - qf2_a_values
            qf1_loss = 0.5*(qf1_error.pow(2) * sample_weights).mean()
            qf2_loss = 0.5*(qf2_error.pow(2) * sample_weights).mean()
            qf_loss = qf1_loss + qf2_loss
            prios = abs(((qf1_error + qf2_error)/2.0 + 1e-5).squeeze())


            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()
            
            rb.update_priorities(sample_batch_inds,sample_env_indices, prios.data.cpu().numpy())
            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            if global_step % 200 == 0 and writer is not None:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), env_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), env_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), env_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), env_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, env_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), env_step)
                writer.add_scalar("losses/alpha", alpha, env_step)
                writer.add_scalar("returns/avg_episodic_return", avg_return/done_count, env_step)
                writer.add_scalar("returns/avg_episodic_length", avg_length/done_count, env_step)
                writer.add_scalar("returns/success_rate", success_count/done_count, env_step)
                writer.add_scalar("charts/SPS", int(env_step / (time.time() - start_time)), env_step)
                print(f"env_step={env_step}, episodic_return={avg_return/done_count}, episodic_length={avg_length/done_count}")
                print("Actor Loss:", actor_loss.item(), "Q Loss:", qf_loss.item() / 2.0, "Alpha:", alpha,end=" ")
                print("Q1 Loss:", qf1_loss.item(), "Q1 Value:", qf1_a_values.mean().item(), "Q2 Value:", qf2_a_values.mean().item(),"\n")
                done_count = avg_return = avg_length = success_count = 0
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), env_step)

            if args.save_model and global_step % args.model_save_interval == 0:
                save_model(actor, qf1, qf2, alpha, global_step, f"runs/{run_name}")
    envs.close()
    eval_envs.close()
    if writer is not None:
        writer.close()