import os
from pathlib import Path
import sys
import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np

from agent.rl.basic_controller import BasicMAC

import collections
from copy import deepcopy
import yaml

from agent.rl.transforms import OneHot


device =  torch.device("cpu")


def get_surrounding(state, width, height, x, y):
    surrounding = [state[(y - 1) % height][x],  # up
                   state[(y + 1) % height][x],  # down
                   state[y][(x - 1) % width],  # left
                   state[y][(x + 1) % width]]  # right

    return surrounding


def make_grid_map(board_width, board_height, beans_positions:list, snakes_positions:dict):
    snakes_map = [[[0] for _ in range(board_width)] for _ in range(board_height)]
    for index, pos in snakes_positions.items():
        for p in pos:
            snakes_map[p[0]][p[1]][0] = index

    for bean in beans_positions:
        snakes_map[bean[0]][bean[1]][0] = 1

    return snakes_map


# Self position:        0:head_x; 1:head_y
# Head surroundings:    2:head_up; 3:head_down; 4:head_left; 5:head_right
# Beans positions:      (6, 7) (8, 9) (10, 11) (12, 13) (14, 15)
# Other snake positions: (16, 17) (18, 19) (20, 21) (22, 23) (24, 25) -- (other_x - self_x, other_y - self_y)
def get_observations(state, agents_index, obs_dim, height, width):
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
    state_ = np.squeeze(state_, axis=2)

    observations = np.zeros((3, obs_dim))
    snakes_position = np.array(snakes_positions_list, dtype=object)
    beans_position = np.array(beans_positions, dtype=object).flatten()
    for i, element in enumerate(agents_index):
        # # self head position
        observations[i][:2] = snakes_positions_list[element][0][:]

        # head surroundings
        head_x = snakes_positions_list[element][0][1]
        head_y = snakes_positions_list[element][0][0]

        head_surrounding = get_surrounding(state_, width, height, head_x, head_y)
        observations[i][2:6] = head_surrounding[:]

        # beans positions
        observations[i][6:16] = beans_position[:]

        # other snake positions
        snake_heads = np.array([snake[0] for snake in snakes_position])
        snake_heads = np.delete(snake_heads, i, 0)
        observations[i][16:] = snake_heads.flatten()[:]
    return observations


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents, args, output_activation='softmax'):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agents = num_agents

        self.args = args

        sizes_prev = [obs_dim, HIDDEN_SIZE]
        sizes_post = [HIDDEN_SIZE, HIDDEN_SIZE, act_dim]

        self.prev_dense = mlp(sizes_prev)
        self.post_dense = mlp(sizes_post, output_activation=output_activation)

    def forward(self, obs_batch):
        out = self.prev_dense(obs_batch)
        out = self.post_dense(out)
        return out


class RLAgent(object):
    def __init__(self, obs_dim, act_dim, num_agent):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agent = num_agent
        self.device = device
        self.output_activation = 'softmax'
        self.actor = Actor(obs_dim, act_dim, num_agent, self.output_activation).to(self.device)

    def choose_action(self, obs):
        obs = torch.Tensor([obs]).to(self.device)
        logits = self.actor(obs).cpu().detach().numpy()[0]
        return logits

    def select_action_to_env(self, obs, ctrl_index):
        logits = self.choose_action(obs)
        actions = logits2action(logits)
        action_to_env = to_joint_action(actions, ctrl_index)
        return action_to_env

    def load_model(self, filename):
        self.actor.load_state_dict(torch.load(filename))


def to_joint_action(action, ctrl_index):
    joint_action_ = []
    action_a = action[ctrl_index]
    each = [0] * 4
    each[action_a] = 1
    joint_action_.append(each)
    return joint_action_


def logits2action(logits):
    logits = torch.Tensor(logits).to(device)
    actions = np.array([Categorical(out).sample().item() for out in logits])
    return np.array(actions)


# agent = RLAgent(26, 4, 3)
# actor_net = os.path.dirname(os.path.abspath(__file__)) + "/actor_2000.pth"
# agent.load_model(actor_net)



def _get_config(config_name):
    
    with open(os.path.join(os.path.dirname(__file__), "{}.yaml".format(config_name)), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict
def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


with open(os.path.join(os.path.dirname(__file__), "default.yaml"), "r") as f:
    try:
        config_dict = yaml.load(f)
    except yaml.YAMLError as exc:
        assert False, "default.yaml error: {}".format(exc)
# Load algorithm and env base configs
env_config = _get_config("snake")
alg_config = _get_config("qmix")
# config_dict = {**config_dict, **env_config, **alg_config}
config_dict = recursive_dict_update(config_dict, env_config)
config_dict = recursive_dict_update(config_dict, alg_config)

_config = config_copy(config_dict)
from types import SimpleNamespace as SN
args = SN(**_config)


env_info = {"state_shape": 78,
                    "obs_shape": 26,
                    "n_actions": 4,
                    "n_agents": 3,
                    "episode_limit": 200}

args.n_agents = env_info["n_agents"]
args.n_actions = env_info["n_actions"]
args.state_shape = env_info["state_shape"]
# print("args: ", args)


# Default/Base scheme
scheme = {
    "state": {"vshape": env_info["state_shape"]},
    "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
    "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
    "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": torch.int},
    "reward": {"vshape": (1,)},
    "terminated": {"vshape": (1,), "dtype": torch.uint8},
    'actions_onehot': {'vshape': (4,), 'dtype': torch.float32, 'group': 'agents'}, 
    'filled': {'vshape': (1,), 'dtype': torch.int64},
}
groups = {
    "agents": args.n_agents
}


# print(scheme)
controller = BasicMAC(scheme, groups, args)
controller.load_models("agent/rl")

# def __init__(self, obs_dim, act_dim, num_agent):
#     self.obs_dim = obs_dim
#     self.act_dim = act_dim
#     self.num_agent = num_agent
#     self.device = device
#     self.output_activation = 'softmax'
#     self.actor = Actor(obs_dim, act_dim, num_agent, self.output_activation).to(self.device)


# # logits  --> (1 ,3, 4)
# def select_action_to_env(logits, obs, ctrl_index):
#     # logits = self.choose_action(obs)
#     actions = logits2action(logits)
#     action_to_env = to_joint_action(actions, ctrl_index)
#     return action_to_env

def encode(action_value):
    aa = [0 for i in range(4)]
    aa[action_value] = 1
    # print(aa)
    return aa


def my_controller(observation_list, action_space_list, is_act_continuous):
    obs_dim = 26
    obs = observation_list.copy()
    board_width = obs['board_width']
    board_height = obs['board_height']
    o_index = obs['controlled_snake_index']  # 2, 3, 4, 5, 6, 7 -> indexs = [0,1,2,3,4,5]
    
    print("--------this is for snake--------- ", o_index-2)
    print(obs)
    if obs['last_direction']==None:
        last_step_actions = torch.zeros(1,3,4)
    else:
        # 'last_direction': ['right', 'right', 'right', 'left', 'up', 'down']
        last_step_str = obs['last_direction'][:3]
        last_step_a = []
        for i in range(3):
            a_str = last_step_str[i]
            # self.actions_name = {-2: "up", 2: "down", -1: "left", 1: "right"}
            #  print("请输入%d个玩家的动作方向[0-3](上下左右)，空格隔开：" % self.n_player)
            if a_str == 'up':
                a_val = 0
            elif a_str == 'down':
                a_val = 1
            elif a_str == 'left':
                a_val = 2
            elif a_str == 'right':
                a_val = 3
                    
            last_step_a.append(a_val)
        print(last_step_str)
        print("last_step_a: ", last_step_a)
        
        last_step_encoded = [encode(step_a) for step_a in last_step_a]
        last_step_actions = torch.tensor(last_step_encoded)
        
        # last_step_actions ----> torch.Size([1, 3, 4])
    
    print("last_step_actions", last_step_actions, last_step_actions.size())

    o_indexs_min = 3 if o_index > 4 else 0
    indexs = [o_indexs_min, o_indexs_min+1, o_indexs_min+2]
    observation = get_observations(obs, indexs, obs_dim, height=board_height, width=board_width)
    # (3, obs_dim) --> (3, 26)
    inputs = torch.tensor([observation])
    print(inputs.shape)
    # inputs-->torch.Size([1, 3, 26])  --->> batch["obs"][:, t] --> (1, 3, 26)
    actions = controller.select_actions(inputs, last_step_actions, test_mode=True)
    # print("!!!!", actions.size())  !!!! torch.Size([3])
    actions = actions.cpu().detach().numpy()
    # print(actions, type(actions)) # [1 1 1] <class 'numpy.ndarray'>
    
    # actions --> torch.Size([3])

    # actions = agent.select_action_to_env(observation, indexs.index(o_index-2))
    # actions = select_action_to_env([actions], observation, indexs.index(o_index-2))
    
    snake_index = o_index-2
    snake_action = actions[snake_index]
    # print([encode(snake_action)])

    return [encode(snake_action)]
    # a list of size 1 4: [[1, 0, 0, 0]]
    #!! GIvcen a fixed snakle index 0 or 1 or 2, output a onehotted vector of the actions for this snake!
