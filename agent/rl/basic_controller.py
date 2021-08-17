# from modules.agents import REGISTRY as agent_REGISTRY
# from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from agent.rl.episode_buffer import EpisodeBatch

agent_REGISTRY = {}
from agent.rl.rnn_agent import RNNAgent
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
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

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
