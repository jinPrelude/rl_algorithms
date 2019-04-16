import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from memory import Memory
from torch.autograd import Variable
import random
from utils import preprocess

class QRDQN_Model(nn.Module) :
    def __init__(self, state_dim, action_dim, num_quant):
        nn.Module.__init__(self)

        self.num_quant = num_quant
        self.action_dim = action_dim

        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, action_dim * num_quant)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.tanh(x)
        x = self.layer2(x)
        return x.view(-1, self.action_dim, self.num_quant)

class QRDQN() :
    def __init__(self, args, state_dim, action_dim, device):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.memory = Memory(args['memory_size'])
        self.model = QRDQN_Model(state_dim, action_dim, args['num_quant']).to(self.device)
        self.target_model = QRDQN_Model(state_dim, action_dim, args['num_quant']).to(self.device)
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=args['lr'])
        self.tau = torch.Tensor((2 * np.arange(args['num_quant']) + 1) / (2.0 * args['num_quant'])).view(1, -1).to(self.device)

        self.target_model.load_state_dict(self.model.state_dict())
        self.epsilon = 1.0

        self.args = args


    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon :
            self.epsilon *= 0.999
            return random.randrange(self.action_dim)
        else :
            state = torch.Tensor(state).to(self.device)
            action = self.model(state).mean(2).max(1)[1]
            return int(action)

    def get_real_action(self, state):
        state = torch.Tensor(state).to(self.device)
        action = self.model(state).mean(2).max(1)[1]
        return int(action)

    def save(self, state, action, reward, next_state ,done):
        self.memory.add((state, action, reward, next_state, done))

    def huber(self, x, k=1.0):
        return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))

    def train(self):
        batch_size = self.args['batch_size']
        mini_batch = self.memory.sample(batch_size)
        mini_batch = np.array(mini_batch).transpose()

        states = np.vstack(mini_batch[0])
        actions = list(mini_batch[1])
        rewards = list(mini_batch[2])
        next_states = np.vstack(mini_batch[3])
        dones = mini_batch[4].astype(int)

        states = torch.Tensor(states).to(self.device)
        next_states = torch.Tensor(next_states).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        rewards = torch.unsqueeze(rewards, 1)
        dones = torch.Tensor(dones).to(self.device)
        dones = torch.unsqueeze(dones, 1)

        states = Variable(states).float()

        # one-hot encoding
        a = torch.LongTensor(actions)

        net_action = self.model(states)
        net_action = net_action[np.arange(batch_size), actions]

        target_action = self.target_model(next_states).detach()
        target_select = target_action.mean(2).max(1)
        target_action = target_action[np.arange(batch_size), target_select[1]]
        target = rewards + (1 - dones) * self.args['gamma'] * target_action
        target = target.t().unsqueeze(-1)

        diff = target - net_action
        loss = self.huber(diff) * (self.tau - (diff.detach() < 0).float()).abs()
        loss = loss.mean()

        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()