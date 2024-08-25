
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Sequential(
                    layer_init(nn.Linear(np.array(env.unwrapped.single_observation_space.shape).prod() + np.prod(env.unwrapped.single_action_space.shape), 256)),
                    nn.Tanh(),
                    layer_init(nn.Linear(256, 256)),
                    nn.Tanh(),
                    layer_init(nn.Linear(256, 256)),
                    nn.Tanh(),
                    layer_init(nn.Linear(256, 1)),
                )
    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.fc1(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Sequential(
            layer_init(nn.Linear(np.array(env.unwrapped.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
        )
        self.fc_mean = nn.Linear(256, np.prod(env.unwrapped.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.unwrapped.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.unwrapped.single_action_space.high - env.unwrapped.single_action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.unwrapped.single_action_space.high+ env.unwrapped.single_action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = self.fc1(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
    
def save_model(actor,qf1,qf2,alpha,global_step,path):
    torch.save({
    'global_step': global_step,
    'actor_state_dict': actor.state_dict(),
    'qf1_state_dict': qf1.state_dict(),
    'qf2_state_dict': qf2.state_dict(),
    'alpha': alpha}, path + f'/model_{global_step}.pt')

def load_model(actor,qf1,qf2,qf1_target,qf2_target,alpha,path):
    checkpoint = torch.load(path)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    qf1.load_state_dict(checkpoint['qf1_state_dict'])
    qf2.load_state_dict(checkpoint['qf2_state_dict'])
    qf1_target.load_state_dict(checkpoint['qf1_state_dict'])
    qf2_target.load_state_dict(checkpoint['qf2_state_dict'])
    alpha = checkpoint['alpha']
    return actor,qf1,qf2,qf1_target,qf2_target,alpha