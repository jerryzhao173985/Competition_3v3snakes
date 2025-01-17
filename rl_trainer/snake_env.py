import gym

from multiagentenv import MultiAgentEnv

import numpy as np

import argparse
import datetime

from tensorboardX import SummaryWriter
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from algo.ddpg import DDPG
from common import *
from log_path import *
from env.chooseenv import make

class SnakeEnv(MultiAgentEnv):

    def __init__(self, **kwargs):
        self.episode_limit = 200
        self.base_env = make('snakes_3v3', conf=None)
        self.current_observations = self.base_env.reset()
        self.n_agents = 3
        self.n_actions = 4 # 4 discrete actions
        print("successfully initialised!")

    def step(self, action_n):
        """ Returns reward, terminated, info """
        # print("action: ", action_n)
        action_random = torch.randint(4,(3,),dtype=torch.long)
        actions = torch.cat((action_n,action_random))
        # print("concatenated: ", actions)
        self.current_observations, rewards, dones, _, infos = self.base_env.step(self.base_env.encode(actions))
        print("reward: ", rewards[:3])
        print("terminated: ", dones)
        # print("observation: ", self.get_obs())
        return np.sum(rewards[:3]),dones, {}

    # Self position:        0:head_x; 1:head_y
    # Head surroundings:    2:head_up; 3:head_down; 4:head_left; 5:head_right
    # Beans positions:      (6, 7) (8, 9) (10, 11) (12, 13) (14, 15)
    # Other snake positions: (16, 17) (18, 19) (20, 21) (22, 23) (24, 25) -- (other_x - self_x, other_y - self_y)
    def get_obs(self, agents_index=[0,1,2], obs_dim=26, height=10, width=20):
        state = self.current_observations[0]
        state_copy = state.copy()
        board_width = state_copy['board_width']
        board_height = state_copy['board_height']
        beans_positions = state_copy[1]
        snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
        snakes_positions_list = []
        for key, value in snakes_positions.items():
            snakes_positions_list.append(value)
        snake_map = make_grid_map(board_width, board_height, beans_positions, snakes_positions)
        state_ = np.array(snake_map)
        state = np.squeeze(state_, axis=2)

        observations = np.zeros((3, obs_dim))
        snakes_position = np.array(snakes_positions_list, dtype=object)
        beans_position = np.array(beans_positions, dtype=object).flatten()
        for i in agents_index:
            # self head position
            observations[i][:2] = snakes_position[i][0][:]

            # head surroundings
            head_x = snakes_position[i][0][1]
            head_y = snakes_position[i][0][0]
            head_surrounding = get_surrounding(state, width, height, head_x, head_y)
            observations[i][2:6] = head_surrounding[:]

            # beans positions
            observations[i][6:16] = beans_position[:]

            # other snake positions
            snake_heads = np.array([snake[0] for snake in snakes_position])
            snake_heads = np.delete(snake_heads, i, 0)
            observations[i][16:] = snake_heads.flatten()[:]
        return observations

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.get_obs()[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        shape = len(self.get_obs_agent(0))
        return shape

    def get_state(self):
        return np.asarray(self.get_obs()).flatten()

    def get_state_size(self):
        """ Returns the shape of the state"""
        shape = len(self.get_state())
        return shape

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones(self.n_actions)

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # there are 9 discrete actions in macad -- refer to macad_gym/core/vehicle_manager.py
        return self.n_actions

    def reset(self):
        """ Returns initial observations and states"""
        try:
            print("resetting")
            self.current_observations = self.base_env.reset()
        except:
            # retry if it doens't work
            print("retrying")
            self.base_env.close()
            self.base_env = make("snakes_3v3", conf=None)
            self.current_observations = self.base_env.reset()
        print("successfully started environment!")
        return self.get_obs(), self.get_state()

    def render(self):
        pass

    def close(self):
        self.base_env.close()

    def seed(self):
        pass

    def save_replay(self):
        pass

# if __name__ == "__main__":
#     env = SnakeEnv()
#     base_env = env.base_env

#     for episode in episodes(n=100):
#         observations = env.reset()

#         dones = {"__all__": False}
#         while not np.all(dones.values()):
#             observations, rewards, dones, infos = env.step([0,1])
#             # episode.record_step(observations, rewards, dones, infos)

if __name__ == "__main__":
    env = SnakeEnv()
    base_env = env.base_env

    for episode in range(1):
        observations = env.reset()
        # print("Initial observations: ", observations[0])

        # dones = {"__all__": False}
        dones = False
        
        # while not np.all(dones.values()):
        while not dones:

            # print("dones", np.all(dones.values()))

            # joint_action = get_join_actions(state, algo_list)

            # random_agents_actions = random_action(3)
            # print(random_agents_actions)

            rl_agents_actions  = torch.Tensor([2,2,2])   #{-2: "up", 2: "down", -1: "left", 1: "right"} 

            # actions_list = rl_agents_actions + list(random_agents_actions)
            # print(actions_list)

            # Intuitively model the actions
            #        state[(y - 1) % height][x],  # up
            #        state[(y + 1) % height][x],  # down
            #        state[y][(x - 1) % width],  # left
            #        state[y][(x + 1) % width]  # right
            rewards, dones, infos = env.step(rl_agents_actions)
            print(env.get_state_size(), env.get_obs_size(), env.get_total_actions(), env.n_agents, env.episode_limit)
            # print("Every step: ", observations)
            # Controlled agent index: o_indexs_min = 3 if o_index > 4 else 0
                                    # indexs = [o_indexs_min, o_indexs_min+1, o_indexs_min+2]
            # episode.record_step(observations, rewards, dones, infos)


        print("episode %d finished!" % (episode+1))
        # print('%s %d' % (name, number))
