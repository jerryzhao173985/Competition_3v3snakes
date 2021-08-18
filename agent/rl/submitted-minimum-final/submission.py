import os
from pathlib import Path
import sys
import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np



#utils
#----------------------------------------------------------------
import torch as th


class Transform:
    def transform(self, tensor):
        raise NotImplementedError

    def infer_output_info(self, vshape_in, dtype_in):
        raise NotImplementedError


class OneHot(Transform):
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def transform(self, tensor):
        y_onehot = tensor.new(*tensor.shape[:-1], self.out_dim).zero_()
        y_onehot.scatter_(-1, tensor.long(), 1)
        return y_onehot.float()

    def infer_output_info(self, vshape_in, dtype_in):
        return (self.out_dim,), th.float32


import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # print("RNN Agent inputs size: ", inputs.size())
        x = F.relu(self.fc1(inputs))
        if hidden_state !=None:
            h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        else:
            h_in = hidden_state
        
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h



#episodeBuffer------EpisodeBatch

import torch as th
import numpy as np
from types import SimpleNamespace as SN


class EpisodeBatch:
    def __init__(self,
                 scheme,
                 groups,
                 batch_size,
                 max_seq_length,
                 data=None,
                 preprocess=None,
                 device="cpu"):
        self.scheme = scheme.copy()
        self.groups = groups
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.preprocess = {} if preprocess is None else preprocess
        self.device = device

        if data is not None:
            self.data = data
        else:
            self.data = SN()
            self.data.transition_data = {}
            self.data.episode_data = {}
            self._setup_data(self.scheme, self.groups, batch_size, max_seq_length, self.preprocess)

    def _setup_data(self, scheme, groups, batch_size, max_seq_length, preprocess):
        if preprocess is not None:
            for k in preprocess:
                assert k in scheme
                new_k = preprocess[k][0]
                transforms = preprocess[k][1]

                vshape = self.scheme[k]["vshape"]
                dtype = self.scheme[k]["dtype"]
                for transform in transforms:
                    vshape, dtype = transform.infer_output_info(vshape, dtype)

                self.scheme[new_k] = {
                    "vshape": vshape,
                    "dtype": dtype
                }
                if "group" in self.scheme[k]:
                    self.scheme[new_k]["group"] = self.scheme[k]["group"]
                if "episode_const" in self.scheme[k]:
                    self.scheme[new_k]["episode_const"] = self.scheme[k]["episode_const"]

        assert "filled" not in scheme, '"filled" is a reserved key for masking.'
        scheme.update({
            "filled": {"vshape": (1,), "dtype": th.long},
        })

        for field_key, field_info in scheme.items():
            assert "vshape" in field_info, "Scheme must define vshape for {}".format(field_key)
            vshape = field_info["vshape"]
            episode_const = field_info.get("episode_const", False)
            group = field_info.get("group", None)
            dtype = field_info.get("dtype", th.float32)

            if isinstance(vshape, int):
                vshape = (vshape,)

            if group:
                assert group in groups, "Group {} must have its number of members defined in _groups_".format(group)
                shape = (groups[group], *vshape)
            else:
                shape = vshape

            if episode_const:
                self.data.episode_data[field_key] = th.zeros((batch_size, *shape), dtype=dtype, device=self.device)
            else:
                self.data.transition_data[field_key] = th.zeros((batch_size, max_seq_length, *shape), dtype=dtype, device=self.device)

    def extend(self, scheme, groups=None):
        self._setup_data(scheme, self.groups if groups is None else groups, self.batch_size, self.max_seq_length)

    def to(self, device):
        for k, v in self.data.transition_data.items():
            self.data.transition_data[k] = v.to(device)
        for k, v in self.data.episode_data.items():
            self.data.episode_data[k] = v.to(device)
        self.device = device

    def update(self, data, bs=slice(None), ts=slice(None), mark_filled=True):
        slices = self._parse_slices((bs, ts))
        for k, v in data.items():
            if k in self.data.transition_data:
                target = self.data.transition_data
                if mark_filled:
                    target["filled"][slices] = 1
                    mark_filled = False
                _slices = slices
            elif k in self.data.episode_data:
                target = self.data.episode_data
                _slices = slices[0]
            else:
                raise KeyError("{} not found in transition or episode data".format(k))

            dtype = self.scheme[k].get("dtype", th.float32)
            v = th.tensor(v, dtype=dtype, device=self.device)
            #self._check_safe_view(v, target[k][_slices])
            target[k][_slices] = v.view_as(target[k][_slices])

            if k in self.preprocess:
                new_k = self.preprocess[k][0]
                v = target[k][_slices]
                for transform in self.preprocess[k][1]:
                    v = transform.transform(v)
                target[new_k][_slices] = v.view_as(target[new_k][_slices])

    def _check_safe_view(self, v, dest):
        idx = len(v.shape) - 1
        for s in dest.shape[::-1]:
            if v.shape[idx] != s:
                if s != 1:
                    raise ValueError("Unsafe reshape of {} to {}".format(v.shape, dest.shape))
            else:
                idx -= 1

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.data.episode_data:
                return self.data.episode_data[item]
            elif item in self.data.transition_data:
                return self.data.transition_data[item]
            else:
                raise ValueError
        elif isinstance(item, tuple) and all([isinstance(it, str) for it in item]):
            new_data = self._new_data_sn()
            for key in item:
                if key in self.data.transition_data:
                    new_data.transition_data[key] = self.data.transition_data[key]
                elif key in self.data.episode_data:
                    new_data.episode_data[key] = self.data.episode_data[key]
                else:
                    raise KeyError("Unrecognised key {}".format(key))

            # Update the scheme to only have the requested keys
            new_scheme = {key: self.scheme[key] for key in item}
            new_groups = {self.scheme[key]["group"]: self.groups[self.scheme[key]["group"]]
                          for key in item if "group" in self.scheme[key]}
            ret = EpisodeBatch(new_scheme, new_groups, self.batch_size, self.max_seq_length, data=new_data, device=self.device)
            return ret
        else:
            item = self._parse_slices(item)
            new_data = self._new_data_sn()
            for k, v in self.data.transition_data.items():
                new_data.transition_data[k] = v[item]
            for k, v in self.data.episode_data.items():
                new_data.episode_data[k] = v[item[0]]

            ret_bs = self._get_num_items(item[0], self.batch_size)
            ret_max_t = self._get_num_items(item[1], self.max_seq_length)

            ret = EpisodeBatch(self.scheme, self.groups, ret_bs, ret_max_t, data=new_data, device=self.device)
            return ret

    def _get_num_items(self, indexing_item, max_size):
        if isinstance(indexing_item, list) or isinstance(indexing_item, np.ndarray):
            return len(indexing_item)
        elif isinstance(indexing_item, slice):
            _range = indexing_item.indices(max_size)
            return 1 + (_range[1] - _range[0] - 1)//_range[2]

    def _new_data_sn(self):
        new_data = SN()
        new_data.transition_data = {}
        new_data.episode_data = {}
        return new_data

    def _parse_slices(self, items):
        parsed = []
        # Only batch slice given, add full time slice
        if (isinstance(items, slice)  # slice a:b
            or isinstance(items, int)  # int i
            or (isinstance(items, (list, np.ndarray, th.LongTensor, th.cuda.LongTensor)))  # [a,b,c]
            ):
            items = (items, slice(None))

        # Need the time indexing to be contiguous
        if isinstance(items[1], list):
            raise IndexError("Indexing across Time must be contiguous")

        for item in items:
            #TODO: stronger checks to ensure only supported options get through
            if isinstance(item, int):
                # Convert single indices to slices
                parsed.append(slice(item, item+1))
            else:
                # Leave slices and lists as is
                parsed.append(item)
        return parsed

    def max_t_filled(self):
        return th.sum(self.data.transition_data["filled"], 1).max(0)[0]

    def __repr__(self):
        return "EpisodeBatch. Batch Size:{} Max_seq_len:{} Keys:{} Groups:{}".format(self.batch_size,
                                                                                     self.max_seq_length,
                                                                                     self.scheme.keys(),
                                                                                     self.groups.keys())


class ReplayBuffer(EpisodeBatch):
    def __init__(self, scheme, groups, buffer_size, max_seq_length, preprocess=None, device="cpu"):
        super(ReplayBuffer, self).__init__(scheme, groups, buffer_size, max_seq_length, preprocess=preprocess, device=device)
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.buffer_index = 0
        self.episodes_in_buffer = 0

    def insert_episode_batch(self, ep_batch):
        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
            self.update(ep_batch.data.transition_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size),
                        slice(0, ep_batch.max_seq_length),
                        mark_filled=False)
            self.update(ep_batch.data.episode_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))
            self.buffer_index = (self.buffer_index + ep_batch.batch_size)
            self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
            self.buffer_index = self.buffer_index % self.buffer_size
            assert self.buffer_index < self.buffer_size
        else:
            buffer_left = self.buffer_size - self.buffer_index
            self.insert_episode_batch(ep_batch[0:buffer_left, :])
            self.insert_episode_batch(ep_batch[buffer_left:, :])

    def can_sample(self, batch_size):
        return self.episodes_in_buffer >= batch_size

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        if self.episodes_in_buffer == batch_size:
            return self[:batch_size]
        else:
            # Uniform sampling only atm
            ep_ids = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)
            return self[ep_ids]

    def __repr__(self):
        return "ReplayBuffer. {}/{} episodes. Keys:{} Groups:{}".format(self.episodes_in_buffer,
                                                                        self.buffer_size,
                                                                        self.scheme.keys(),
                                                                        self.groups.keys())


#----------------------------------------------------------------

#basic_controller script
#----------------------------------------------------------------

# from modules.agents import REGISTRY as agent_REGISTRY
# from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th

agent_REGISTRY = {}

agent_REGISTRY["rnn"] = RNNAgent

# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type
        self.epsilon = 0.0

        # self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def action_selector(self, agent_inputs, avail_actions, test_mode=True):
        # agent_inputs --> torch.Size([3, 4]) , avail_actions --> torch.Size([3, 4])

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        # self.epsilon = self.schedule.eval(t_env)

        # if test_mode:
            # Greedy action selection only
        self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!
        
        # agent_inputs --> torch.Size([3, 4]) , random_numbers--> torch.Size([3])
        
        # random_numbers = th.rand_like(agent_inputs[:, 0])
        # print(random_numbers.size())

        # pick_random = (random_numbers < self.epsilon).long()
        # print(pick_random)

        # random_actions = Categorical(avail_actions.float()).sample().long()

        # picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]

        # print(masked_q_values)
        # print(masked_q_values.max(dim=1))
#         torch.return_types.max(
# values=tensor([1.1650, 1.0956, 1.1531], grad_fn=<MaxBackward0>),
# indices=tensor([1, 2, 1]))

        picked_actions = masked_q_values.max(dim=1)[1]     #get indices of torch.max()
        # print("piccked: " ,picked_actions)
        # print("picked siize: ", picked_actions.size())

        return picked_actions
        # picked siize:  torch.Size([3])

    def select_actions(self, inputs, actions, test_mode=True):
        # last step actions:  torch.Size([1, 3, 4])

        # inputs-->torch.Size([1, 3, 26])  --->> batch["obs"][:, t] --> (1, 3, 26)

        # Only select actions for the selected batch elements in bs
        avail_actions = th.tensor([[1, 1, 1, 1],[1, 1, 1, 1],[1, 1, 1, 1]], dtype = th.int32)
        
        # agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        agent_inputs = self._build_inputs(inputs, actions)
        # agent_inputs --> torch.Size([3, 33])

        # avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outs, self.hidden_states = self.agent(agent_inputs.float(), self.hidden_states)

        # print("agent_out 1: ", agent_outs.size())

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(1 * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        # agent_outputs = agent_outs.view(1, self.n_agents, -1)
        # print("agent_out 2: ", agent_outs.size())
        # print("avail_actions : ", avail_actions.size())

        chosen_actions = self.action_selector(agent_outs, avail_actions, test_mode=test_mode)
        # agent_outs --> torch.Size([3, 4]) , avail_actions --> torch.Size([3, 4])

        return chosen_actions
        # torch.Size([3])


    # def forward(self, ep_batch, t_ep, test_mode=True):
    #     agent_inputs = self._build_inputs(ep_batch, t_ep)
    #     avail_actions = ep_batch["avail_actions"][:, t_ep]
    #     agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

    #     # Softmax the agent outputs if they're policy logits
    #     if self.agent_output_type == "pi_logits":

    #         if getattr(self.args, "mask_before_softmax", True):
    #             # Make the logits for unavailable actions very negative to minimise their affect on the softmax
    #             reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
    #             agent_outs[reshaped_avail_actions == 0] = -1e10

    #         agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

    #     return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)





    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.pth".format(path))

    def load_models(self):
        self.agent.load_state_dict(th.load(os.path.dirname(os.path.abspath(__file__)) +"/agent.pth", map_location=torch.device('cpu')))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, input, actions):   # batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent

        # bs = batch.batch_size
        bs = 1
        inputs = []
        # inputs.append(batch["obs"][:, t])  # b1av
        inputs.append(input)

        # print("------",batch["obs"].cpu().detach().numpy().shape, batch["obs"][:, t].cpu().detach().numpy().shape)
        # batch["obs"]-->(1, 201, 3, 26)
        # batch["obs"][:, t] --> (1, 3, 26)

        # input0:  2 torch.Size([1, 3, 4])
        # input1:  3 torch.Size([1, 3, 3])
#         inputs.size:   torch.Size([3, 33])
# rnn input shape:  torch.Size([3, 33])
# output action shape: torch.Size([3, 4])

        # if self.args.obs_last_action:
        #     if t == 0:
        #         inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
        #     else:
        #         inputs.append(batch["actions_onehot"][:, t-1])

        # Append Last step onehot actions
        # input0:  2 torch.Size([1, 3, 4])
        inputs.append(actions)

        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents).unsqueeze(0).expand(bs, -1, -1))
        
        # print(inputs[2])

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        
        # print(inputs.size())

        # rnn agent network inputs size:   torch.Size([3, 33])
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape






#----------------------------------------------------------------
#Main Script
#----------------------------------------------------------------


import collections
from copy import deepcopy
#import yaml


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


# class Actor(nn.Module):
#     def __init__(self, obs_dim, act_dim, num_agents, args, output_activation='softmax'):
#         super().__init__()

#         self.obs_dim = obs_dim
#         self.act_dim = act_dim
#         self.num_agents = num_agents

#         self.args = args

#         sizes_prev = [obs_dim, HIDDEN_SIZE]
#         sizes_post = [HIDDEN_SIZE, HIDDEN_SIZE, act_dim]

#         self.prev_dense = mlp(sizes_prev)
#         self.post_dense = mlp(sizes_post, output_activation=output_activation)

#     def forward(self, obs_batch):
#         out = self.prev_dense(obs_batch)
#         out = self.post_dense(out)
#         return out


# class RLAgent(object):
#     def __init__(self, obs_dim, act_dim, num_agent):
#         self.obs_dim = obs_dim
#         self.act_dim = act_dim
#         self.num_agent = num_agent
#         self.device = device
#         self.output_activation = 'softmax'
#         self.actor = Actor(obs_dim, act_dim, num_agent, self.output_activation).to(self.device)

#     def choose_action(self, obs):
#         obs = torch.Tensor([obs]).to(self.device)
#         logits = self.actor(obs).cpu().detach().numpy()[0]
#         return logits

#     def select_action_to_env(self, obs, ctrl_index):
#         logits = self.choose_action(obs)
#         actions = logits2action(logits)
#         action_to_env = to_joint_action(actions, ctrl_index)
#         return action_to_env

#     def load_model(self, filename):
#         self.actor.load_state_dict(torch.load(filename))


# def to_joint_action(action, ctrl_index):
#     joint_action_ = []
#     action_a = action[ctrl_index]
#     each = [0] * 4
#     each[action_a] = 1
#     joint_action_.append(each)
#     return joint_action_


# def logits2action(logits):
#     logits = torch.Tensor(logits).to(device)
#     actions = np.array([Categorical(out).sample().item() for out in logits])
#     return np.array(actions)


# agent = RLAgent(26, 4, 3)
# actor_net = os.path.dirname(os.path.abspath(__file__)) + "/actor_2000.pth"
# agent.load_model(actor_net)


#
#def _get_config(config_name):
#
#    with open(os.path.join(os.path.dirname(__file__), "{}.yaml".format(config_name)), "r") as f:
#        try:
#            config_dict = yaml.load(f)
#        except yaml.YAMLError as exc:
#            assert False, "{}.yaml error: {}".format(config_name, exc)
#        return config_dict
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


#with open(os.path.join(os.path.dirname(__file__), "default.yaml"), "r") as f:
#    try:
#        config_dict = yaml.load(f)
#    except yaml.YAMLError as exc:
#        assert False, "default.yaml error: {}".format(exc)

#print(config_dict)
# Load algorithm and env base configs
#env_config = _get_config("snake")
#alg_config = _get_config("qmix")
# config_dict = {**config_dict, **env_config, **alg_config}
#config_dict = recursive_dict_update(config_dict, env_config)
#config_dict = recursive_dict_update(config_dict, alg_config)


#manually write the config here instead off the yaml file
config_dict = {

        # --- Defaults ---

    # --- pymarl options ---
        "runner": "episode", # Runs 1 env for an episode
        "mac": "basic_mac", # Basic controller
        "env": "sc2", # Environment name
        "env_args": {}, # Arguments for the environment
        "batch_size_run": 1, # Number of environments to run in parallel
        "test_nepisode": 20, # Number of episodes to test for
        "test_interval": 2000, # Test after {} timesteps have passed
        "test_greedy": True, # Use greedy evaluation (if False, will set epsilon floor to 0
        "log_interval": 2000, # Log summary of stats after every {} timesteps
        "runner_log_interval": 200, # Log runner stats (not test stats) every {} timesteps
        "learner_log_interval": 200, # Log training stats every {} timesteps
        "t_max": 10000, # Stop running after this many timesteps
        "use_cuda": True, # Use gpu by default unless it isn't available
        "buffer_cpu_only": True, # If true we won't keep all of the replay buffer in vram

    # --- Logging options ---
        "use_tensorboard": False, # Log results to tensorboard
        "save_model": False, # Save the models to disk
        "save_model_interval": 2000000, # Save models after this many timesteps
        "checkpoint_path": "", # Load a checkpoint from this path
        "evaluate": False, # Evaluate model for test_nepisode episodes and quit (no training)
        "load_step": 0, # Load model trained on this many timesteps (0 if choose max possible)
        "save_replay": False, # Saving the replay of the model loaded from checkpoint_path
        "local_results_path": "results", # Path for local results

    # --- RL hyperparameters ---
        "gamma": 0.99,
        "batch_size": 32, # Number of episodes to train on
        "buffer_size": 32, # Size of the replay buffer
        "lr": 0.0005, # Learning rate for agents
        "critic_lr": 0.0005, # Learning rate for critics
        "optim_alpha": 0.99, # RMSProp alpha
        "optim_eps": 0.00001, # RMSProp epsilon
        "grad_norm_clip": 10, # Reduce magnitude of gradients above this L2 norm

    # --- Agent parameters ---
        "agent": "rnn", # Default rnn agent
        "rnn_hidden_dim": 64, # Size of hidden state for default rnn agent
        "obs_agent_id": True, # Include the agent's one_hot id in the observation
        "obs_last_action": True, # Include the agent's last action (one_hot) in the observation

    # --- Experiment running params ---
        "repeat_id": 1,
        "label": "default_label",




    #From other files
    # --- QMIX specific parameters ---

    # use epsilon greedy action selector
        "action_selector": "epsilon_greedy",
        "epsilon_start": 1.0,
        "epsilon_finish": 0.001,
        "epsilon_anneal_time": 1000000,

        "grad_norm_clip": 10,

        "runner": "episode",

        "buffer_size": 1000,

    # update the target network every {} episodes
        "target_update_interval": 100,

    # use the Q_Learner to train
        "agent_output_type": "q",
        "learner": "q_learner",
        "double_q": True,
        "mixer": "qmix",
        "mixing_embed_dim": 32,
        "hypernet_layers": 2,
        "hypernet_embed": 64,

        "name": "qmix",



        "env": "snake",

        "env_args":{
          "seed": 10,
          "episode_limit": 200},

        "save_model": True,
    #checkpoint_path: ""
        "save_model_interval": 10000,
        "batch_size": 64,
        "buffer_cpu_only": True,
        "test_greedy": True,
        "test_nepisode": 32,
        "test_interval": 10000,
        "log_interval": 200,
        "runner_log_interval": 200,
        "learner_log_interval": 200,
        "t_max": 2050000,
        "use_tensorboard": True,




}



_config = config_copy(config_dict)
#print("2----: ", _config)
from types import SimpleNamespace as SN
args = SN(**_config)
#print("args:  ", args)


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
#ss = "agent/rl/trained_model/1900200"

#ss = "agent/rl/trained_model/1470200"
#ss = "."
controller.load_models()

#print(ss)
# Checkpoint from 200 to 2050,200

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
    # print(obs)
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
        # print(last_step_str)
        # print("last_step_a: ", last_step_a)
        
        last_step_encoded = [encode(step_a) for step_a in last_step_a]
        last_step_actions = torch.tensor(last_step_encoded)
        
        # last_step_actions ----> torch.Size([1, 3, 4])
    
    # print("last_step_actions", last_step_actions, last_step_actions.size())

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
    
    if o_indexs_min==0:
        snake_index = o_index-2
    elif o_indexs_min==3:
        snake_index = o_index-2-3

    snake_action = actions[snake_index]
    # print([encode(snake_action)])

    return [encode(snake_action)]
    # a list of size 1 4: [[1, 0, 0, 0]]
    #!! GIvcen a fixed snakle index 0 or 1 or 2, output a onehotted vector of the actions for this snake!
