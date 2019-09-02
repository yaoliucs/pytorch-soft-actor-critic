import os
import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, QNetworkDA, DeterministicPolicy, SoftmaxPolicy
from collections import OrderedDict

class SAC(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.discrete_action = args.discrete_action

        if self.discrete_action:
            self.critic = QNetworkDA(num_inputs, action_space.n, args.hidden_size).to(device=self.device)
            self.critic_target = QNetworkDA(num_inputs, action_space.n, args.hidden_size).to(self.device)
        else:
            self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
            self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)

        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning == True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        elif self.policy_type == "Softmax":
            if self.automatic_entropy_tuning == True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.n).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = SoftmaxPolicy(num_inputs, action_space.n, args.hidden_size).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
            self.demon_policy = SoftmaxPolicy(num_inputs, action_space.n, args.hidden_size).to(self.device)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)


    def select_action(self, state, eval=False, demon=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            if demon:
                action, _, _ = self.demon_policy.sample(state)
            else:
                action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        action = action.detach().cpu().numpy()
        if self.discrete_action:
            action = action[0]
        return action[0]


    def update_parameters(self, memory, batch_size, updates):
        ActionTensor = torch.LongTensor if self.discrete_action else torch.FloatTensor
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = ActionTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        if self.discrete_action:
            min_qf_pi = min_qf_pi.detach()
            policy_loss = (log_pi*(0.5*self.alpha*log_pi + self.alpha - min_qf_pi)).mean()
        else:
            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
    
    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

    def load_athens_model(self, model_path):
        sd = torch.load(model_path)
        nsd = OrderedDict()
        nsd['linear1.weight'] = sd['base.mlp.0.weight']
        nsd['linear1.bias'] = sd['base.mlp.0.bias']
        nsd['linear2.weight'] = sd['base.mlp.2.weight']
        nsd['linear2.bias'] = sd['base.mlp.2.bias']
        nsd['linear3.weight'] = sd['base.mlp.4.weight']
        nsd['linear3.bias'] = sd['base.mlp.4.bias']
        nsd['out_linear.weight'] = sd['output_model.0.weight']
        nsd['out_linear.bias'] = sd['output_model.0.bias']
        self.demon_policy.load_state_dict(nsd)

