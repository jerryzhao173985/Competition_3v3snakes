import numpy as np
import torch
import random
from agent.rl.submission import controller, get_observations, encode
from env.chooseenv import make
from tabulate import tabulate
import argparse
from torch.distributions import Categorical
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_actions(state, algo, indexs):

    # random agent
    actions = np.random.randint(4, size=3)

    # rl agent
    if algo == 'rl':

        if state['last_direction']==None:
            last_step_actions = torch.zeros(1,3,4)
        else:
            last_step_str = state['last_direction'][:3]
            last_step_a = []
            for i in range(3):
                a_str = last_step_str[i]
                if a_str == 'up':
                    a_val = 0
                elif a_str == 'down':
                    a_val = 1
                elif a_str == 'left':
                    a_val = 2
                elif a_str == 'right':
                    a_val = 3
                last_step_a.append(a_val)
            
            last_step_encoded = [encode(step_a) for step_a in last_step_a]
            last_step_actions = torch.tensor(last_step_encoded)
            
            
        obs = get_observations(state, indexs, obs_dim=26, height=10, width=20)
        # logits = agent.choose_action(obs)
        # logits = torch.Tensor(logits)
        # actions = np.array([Categorical(out).sample().item() for out in logits])
        # print(actions)

        inputs = torch.tensor([obs])
        actions = controller.select_actions(inputs, last_step_actions, test_mode=True)
        actions = actions.cpu().detach().numpy()
        # print(actions, type(actions)) # [1 1 1] <class 'numpy.ndarray'>

    return actions
    # [0 3 3]


def get_join_actions(obs, algo_list):
    obs_2_evaluation = obs[0]
    indexs = [0,1,2,3,4,5]
    first_action = get_actions(obs_2_evaluation, algo_list[0], indexs[:3])
    second_action = get_actions(obs_2_evaluation, algo_list[1], indexs[3:])
    actions = np.zeros(6)
    actions[:3] = first_action[:]
    actions[3:] = second_action[:]
    return actions


def run_game(env, algo_list, episode, verbose=False):

    total_reward = np.zeros(6)
    num_win = np.zeros(3)

    for i in range(1, episode + 1):
        episode_reward = np.zeros(6)

        state = env.reset()

        step = 0

        while True:
            joint_action = get_join_actions(state, algo_list)

            next_state, reward, done, _, info = env.step(env.encode(joint_action))  # (6,6,bool,_,4)

            # print(len(next_state[0]), reward[0],  len(info))   #(11, 0, 4)
            # print(joint_action)

            reward = np.array(reward)
            episode_reward += reward

            if done:
                if np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]):
                    num_win[0] += 1
                elif np.sum(episode_reward[:3]) < np.sum(episode_reward[3:]):
                    num_win[1] += 1
                else:
                    num_win[2] += 1

                if not verbose:
                    print('.', end='')
                    if i % 100 == 0 or i == episode:
                        print()
                break

            state = next_state
            step += 1

        total_reward += episode_reward

    # calculate results
    total_reward /= episode
    print("total_reward: ", total_reward)
    print(f'\nResult base on {episode} ', end='')
    print('episode:') if episode == 1 else print('episodes:')

    header = ['Name', algo_list[0], algo_list[1]]
    data = [['score', np.round(np.sum(total_reward[:3]), 2), np.round(np.sum(total_reward[3:]), 2)],
            ['win', num_win[0], num_win[1]]]
    print(tabulate(data, headers=header, tablefmt='pretty'))


if __name__ == "__main__":
    env_type = 'snakes_3v3'

    game = make(env_type, conf=None)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--my_ai", default="rl", help="rl/random")
    parser.add_argument("--opponent", default="random", help="rl/random")
    parser.add_argument("--episode", default=100)
    args = parser.parse_args()

    agent_list = [args.my_ai, args.opponent]
    run_game(game, algo_list=agent_list, episode=args.episode, verbose=False)
