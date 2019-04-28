import math
import torch
import gym

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class FixHorizon(gym.Wrapper):
    def __init__(self, env, horizon, self_loop_reward=0):
        super(FixHorizon, self).__init__(env)
        self._max_horizon = horizon
        self._step = 0
        self._last_obs = None
        self._true_done = False
        self._loop_reward = self_loop_reward
        self._max_episode_steps = env._max_episode_steps

    def step(self, action):
        if self._true_done:
            obs, reward, done, info = self.self_loop()
        else:
            obs, reward, done, info = self.env.step(action)
            self._last_obs = obs
            if done:
                self._true_done = True
        self._step += 1
        if self._step >= self._max_horizon:
            done = True
        else:
            done = False
        return obs, reward, done, info

    def self_loop(self):
        return self._last_obs, self._loop_reward, True, None

    def reset(self, **kwargs):
        self._last_obs = self.env.reset()
        self._step = 0
        self._true_done = False
        return self._last_obs
