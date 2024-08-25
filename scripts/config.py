from dataclasses import dataclass

@dataclass
class Args:
    exp_name: str = "TestSAC"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "SAC_imporv"
    """the wandb's project name"""
    wandb_entity: str = "" 
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    model_save_interval: int = 2000
    """the interval to save model in timestep"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: str = ""
    """path to a pretrained checkpoint file to start evaluation/training from"""

    # Algorithm specific arguments
    env_id: str = "PickCube-v1"
    """the environment id of the task"""
    total_timesteps: int = 50000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-3
    """the learning rate of the optimizer"""
    num_envs: int = 512
    """the number of parallel environments"""
    num_eval_envs: int = 8
    """the number of parallel evaluation environments"""
    buffer_size: float = 1e6
    """the replay memory buffer size"""
    partial_reset: bool = True
    """toggle if the environments should perform partial resets"""
    num_steps: int = 400
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 400
    """the number of steps to run in each evaluation environment during evaluation"""

    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.015
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 512
    """the batch size of sample from the reply memory"""
    learning_starts: int = 100
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 6e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 2  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    RESCALE_REWARDS: bool = False
    """if toggled, will rescale the rewardS on done signal"""

    target_kl: float = 0.1
    """the target KL divergence threshold"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    log: bool = True
    """if toggled, will log to tensorboard"""

    # demo only parameters
    use_demo: bool = False
    """if toggled, will use demonstrations"""
    demo_percent: float = 0.5
    """the percentage of the demonstrations to use"""

    # PER parameters
    per_alpha: float = 0.6
    """alpha parameter for PER"""
    per_beta_start: float = 0.4
    """initial beta parameter for PER"""
    per_beta_frames: int =100000
    """number of frames over which beta is annealed"""
    use_max_priority: bool = False
    """if toggled, will use max priority for PER"""
    sd_scale: float = 1.2
    """the scal of the standard deviation added to the PER priorities"""
