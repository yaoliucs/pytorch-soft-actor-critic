import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
import cartpole_continuous
from sac import SAC
from tensorboardX import SummaryWriter
from normalized_actions import NormalizedActions
from replay_memory import ReplayMemory
from utils import FixHorizon

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='name of the environment to run')
parser.add_argument('--policy', default="Gaussian",
                    help='algorithm to use: Gaussian | Deterministic')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default:True)')
parser.add_argument('--n_eval', type=int, default=10,
                    help='Number of episodes used to evaluate policy (default:10)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Temperature parameter α automaically adjusted.')
parser.add_argument('--seed', type=int, default=456, metavar='N',
                    help='random seed (default: 456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--end_steps', type=int, default=10000, metavar='N',
                    help='Total steps sampling actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=10000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--discrete_action', action="store_true",
                    help='Discrete action space')
args = parser.parse_args()

# python main.py --env-name CartPole-v0 --discrete_action --policy "Softmax" --automatic_entropy_tuning True --num_steps 15000 --batch_size 500 --hidden_size 32 --lr 0.001
# python main.py --env-name CartPole-v0 --discrete_action --policy "Softmax" --automatic_entropy_tuning True --num_steps 110000 --batch_size 5000 --hidden_size 32 --lr 0.001 --start_steps 100000 --end_steps 100000 --replay_size 100000

for k, v in vars(args).items():
    print(' ' * 26 + k + ': ' + str(v))

# Environment
# Removing Normalized Actions. 
# Another way to use it = actions * env.action_space.high[0] -> (https://github.com/sfujim/TD3). This does the same thing.
# (or add env._max_episode_steps to normalized_actions.py)
env = gym.make(args.env_name)
env = FixHorizon(env,horizon=200)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)

eval_env = gym.make(args.env_name)
eval_env.seed(args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

#TesnorboardX
writer = SummaryWriter(log_dir='runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))
result = []

# Memory
memory = ReplayMemory(args.replay_size)

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size and total_numsteps >= args.start_steps:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        episode_steps += 1
        total_numsteps += 1
        next_state, reward, done, _ = env.step(action) # Step
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        if args.end_steps > total_numsteps:
            if args.discrete_action:
                action = np.array([action])
            memory.push(state, action, reward, next_state, mask) # Append transition to memory
        state = next_state

        if total_numsteps >= args.start_steps and total_numsteps % 100 == 0 and args.eval == True:
            eval_rewards = []
            for i in range(args.n_eval):
                eval_state = eval_env.reset()
                eval_episode_reward = 0
                eval_done = False
                while not eval_done:
                    eval_action = agent.select_action(eval_state, eval=False)

                    eval_next_state, eval_reward, eval_done, _ = eval_env.step(eval_action)
                    eval_episode_reward += eval_reward

                    eval_state = eval_next_state
                eval_rewards.append(eval_episode_reward)
            eval_episode_reward = np.mean(eval_rewards)

            print("Test Step: {}, reward: {}".format(total_numsteps, round(eval_episode_reward, 2)))
            result.append(eval_episode_reward)

    if total_numsteps > args.num_steps:
        break

    # writer.add_scalar('reward/train', episode_reward, i_episode)
    # print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    # if i_episode % 10 == 0 and args.eval == True:
    #     state = env.reset()
    #     episode_reward = 0
    #     done = False
    #     while not done:
    #         action = agent.select_action(state, eval=True)
    #
    #         next_state, reward, done, _ = env.step(action)
    #         episode_reward += reward
    #
    #
    #         state = next_state
    #
    #
    #     writer.add_scalar('reward/test', episode_reward, i_episode)
    #
    #     print("----------------------------------------")
    #     print("Test Episode: {}, reward: {}".format(i_episode, round(episode_reward, 2)))
    #     print("----------------------------------------")

np.savetxt(X=np.array(result),fname='runs/{}_SAC_{}_{}_{}.txt'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

env.close()

