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
        # self.episode_limit = kwargs['episode_limit']
        self.episode_limit = 200
        self.base_env = make("snakes_3v3", conf=None)
        self.current_observations = self.base_env.reset()
        # FIXME
        # for key in self.current_observations:
        #     self.agent_ids.append(key)
        
        agent_ids = [0, 1, 2, 3, 4, 5]
        self.n_agents = 6 # number of controlling agents + random agent (random way of training/ greedy strategy)
        self.n_actions = 4  # 3 discrete actions in snake -- //{-2: "up", 2: "down", -1: "left", 1: "right"}
        print("successfully initialised!")

    def step(self, all_actions):
        """ Returns reward, terminated, info """
        
        # joint_action = get_join_actions(state, algo_list)
        self.current_observations, rewards, dones, _, infos = self.base_env.step(self.base_env.encode(all_actions))
        # r_n = []
        # d_n = []
        # r_n = rewards

        # d_n = [dones] * 6
        # for agent_id in self.agent_ids:
        #     # r_n.append(rewards.get(agent_id))
        #     d_n.append(dones.get(agent_id, True))
        # done = all(d_n)
        print("reward: ", rewards[:3])
        print("terminated: ", dones)

        return np.sum(rewards[:3]),dones, {}

    def get_obs(self):
        """ Returns all agent observations in a list """
        # obs_n = []
        # for agent_id in self.agent_ids:
        #     obs_n.append(self.current_observations.get(agent_id).flatten())
        return self.current_observations

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.get_obs()[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        shape = len(self.get_obs_agent(0))
        return shape

    def get_state(self):
        # return np.asarray(self.get_obs()).flatten()
        return self.get_obs_agent(0)
        # # During training, since all agents are given the same obs, we take the state of 1st agent.

    def get_state_size(self): # Here this is same as get_obs_size()
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

    for episode in range(100):
        observations = env.reset()

        dones = {"__all__": False}
        while not np.all(dones.values()):

            # joint_action = get_join_actions(state, algo_list)
            actions_list  = [2,2,2,2,2,2]
            observations, rewards, dones, infos = env.step(actions_list)
            # Controlled agent index: o_indexs_min = 3 if o_index > 4 else 0
                                    # indexs = [o_indexs_min, o_indexs_min+1, o_indexs_min+2]
            # episode.record_step(observations, rewards, dones, infos)
