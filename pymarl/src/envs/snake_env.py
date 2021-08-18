import gym
from envs.multiagentenv import MultiAgentEnv
from envs.chooseenv import make
from envs.common import *
from envs.log_path import *
import torch

import numpy as np

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
        print("action: ", action_n)
        action_random = torch.randint(4,(3,),dtype=torch.long)
        actions = torch.cat((action_n,action_random))
        print("concatenated: ", actions)
        self.current_observations, rewards, dones, _, infos = self.base_env.step(self.base_env.encode(actions))
        print("reward: ", rewards[:3])
        print("terminated: ", dones)
        #print("observation: ", self.get_obs())
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

if __name__ == "__main__":
    env = SnakeEnv()
    base_env = env.base_env

    for episode in episodes(n=100):
        observations = env.reset()

        dones = {"__all__": False}
        while not np.all(dones.values()):
            observations, rewards, dones, infos = env.step([0,1])
            # episode.record_step(observations, rewards, dones, infos)
