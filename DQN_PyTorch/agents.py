import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from memory import Memory
from torch.autograd import Variable
import random
from utils import preprocess

class Model(nn.Module) :
    def __init__(self, state_dim, action_dim):
        super(Model, self).__init__()
        self.f1 = nn.Sequential(
            nn.Linear(state_dim, 300),
            nn.Tanh(),
            nn.Linear(300, 100),
            nn.Tanh(),
            nn.Linear(100, action_dim)
        )

    def forward(self, x):
        return self.f1(x)

class Cnn_Model(nn.Module) :
    def __init__(self, action_dim):
        super(Cnn_Model, self).__init__()
        self.f1 = nn.Sequential(
            nn.Conv2d(4, 16, 8, 4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU()
        )
        self.f2 = nn.Sequential(
            nn.Linear(32*9*9, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x, batch_size = 1):
        x = self.f1(x)
        x = x.view(batch_size, -1)
        x = self.f2(x)
        return x

class DQN() :
    def __init__(self, args, state_dim, action_dim, device):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.memory = Memory(args['memory_size'])
        self.model = Model(state_dim, action_dim).to(self.device)
        self.target_model = Model(state_dim, action_dim).to(self.device)
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=args['lr'])

        self.target_model.load_state_dict(self.model.state_dict())
        self.epsilon = 1.0

        self.args = args


    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon :
            self.epsilon -= 0.0000009
            return random.randrange(self.action_dim)
        else :
            state = torch.Tensor(state).to(self.device)
            _, action = torch.max(self.model(state), 0)
            return int(action)

    def get_real_action(self, state):
        state = torch.Tensor(state).to(self.device)
        _, action = torch.max(self.model(state), 0)
        return int(action)

    def save(self, state, action, reward, next_state ,done):
        self.memory.add((state, action, reward, next_state, done))

    def train(self):
        mini_batch = self.memory.sample(self.args['batch_size'])
        mini_batch = np.array(mini_batch).transpose()

        states = np.vstack(mini_batch[0])
        actions = list(mini_batch[1])
        rewards = list(mini_batch[2])
        next_states = np.vstack(mini_batch[3])
        dones = mini_batch[4].astype(int)

        states = torch.Tensor(states).to(self.device)
        next_states = torch.Tensor(next_states).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        dones = torch.Tensor(dones).to(self.device)

        states = Variable(states).float()
        pred = self.model(states)

        # one-hot encoding
        a = torch.LongTensor(actions).view(-1, 1)

        one_hot_action = torch.FloatTensor(self.args['batch_size'], self.action_dim).zero_()
        one_hot_action.scatter_(1, a, 1)
        one_hot_action = torch.Tensor(one_hot_action).to(self.device)

        pred = torch.sum(pred.mul(Variable(one_hot_action)), dim=1)

        next_states = Variable(next_states).float()
        next_pred = self.target_model(next_states)

        # rewards = torch.FloatTensor(rewards)
        # dones = torch.FloatTensor(dones)

        # Q Learning: get maximum Q value at s' from target model
        target = rewards + (1 - dones) * self.args['gamma'] * next_pred.max(1)[0]
        target = Variable(target)

        self.model_optimizer.zero_grad()

        # MSE Loss function
        loss = F.mse_loss(pred, target)
        loss.backward()

        # and train
        self.model_optimizer.step()

class Double_DQN() :
    def __init__(self, args, state_dim, action_dim, device):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.memory = Memory(args['memory_size'])
        self.model = Model(state_dim, action_dim).to(self.device)
        self.target_model = Model(state_dim, action_dim).to(self.device)
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=args['lr'])

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
            _, action = torch.max(self.model(state), 0)
            return int(action)

    def get_real_action(self, state):
        state = torch.Tensor(state).to(self.device)
        _, action = torch.max(self.model(state), 0)
        return int(action)

    def save(self, state, action, reward, next_state ,done):
        self.memory.add((state, action, reward, next_state, done))

    def train(self):
        mini_batch = self.memory.sample(self.args['batch_size'])
        mini_batch = np.array(mini_batch).transpose()

        states = np.vstack(mini_batch[0])
        actions = list(mini_batch[1])
        rewards = list(mini_batch[2])
        next_states = np.vstack(mini_batch[3])
        dones = mini_batch[4].astype(int)

        states = torch.Tensor(states).to(self.device)
        next_states = torch.Tensor(next_states).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        dones = torch.Tensor(dones).to(self.device)

        states = Variable(states).float()
        pred = self.model(states)

        # one-hot encoding
        a = torch.LongTensor(actions).view(-1, 1)

        one_hot_action = torch.FloatTensor(self.args['batch_size'], self.action_dim).zero_()
        one_hot_action.scatter_(1, a, 1)
        one_hot_action = torch.Tensor(one_hot_action).to(self.device)

        pred = torch.sum(pred.mul(Variable(one_hot_action)), dim=1)

        next_states = Variable(next_states).float()
        next_pred = self.target_model(next_states)

        # rewards = torch.FloatTensor(rewards)
        # dones = torch.FloatTensor(dones)

        # Q Learning: get maximum Q value at s' from target model
        q_next_select = self.model(next_states)
        _, q_next_select = torch.max(q_next_select, 0)

        one_hot_action = torch.FloatTensor(self.args['batch_size'], self.action_dim).zero_()
        one_hot_action.scatter_(1, a, 1)
        one_hot_action = torch.Tensor(one_hot_action).to(self.device)

        target_pred = self.target_model(next_states)
        target_pred = torch.sum(target_pred.mul(Variable(one_hot_action)), dim=1)

        target = rewards + (1 - dones) * self.args['gamma'] * target_pred
        target = Variable(target)

        self.model_optimizer.zero_grad()

        # MSE Loss function
        loss = F.mse_loss(pred, target)
        loss.backward()

        # and train
        self.model_optimizer.step()

class Double_DQN_Cnn() :
    def __init__(self, args, state_dim, action_dim, device):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.memory = Memory(args['memory_size'])
        self.model = Cnn_Model(action_dim).to(self.device)
        self.target_model = Cnn_Model(action_dim).to(self.device)
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=args['lr'])

        self.target_model.load_state_dict(self.model.state_dict())
        self.epsilon = 1.0

        self.args = args


    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon :
            if self.epsilon > 0.11 :
                self.epsilon -= 0.0000009
            return random.randrange(self.action_dim)
        else :
            state = torch.Tensor(state).to(self.device)
            _, action = torch.max(self.model(state), 1)
            return int(action)

    def get_real_action(self, state):
        state = torch.Tensor(state).to(self.device)
        _, action = torch.max(self.model(state), 1)
        return int(action)

    def save(self, state, action, reward, next_state ,done):
        self.memory.add((state, action, reward, next_state, done))

    def train(self):
        mini_batch = self.memory.sample(self.args['batch_size']*self.args['skip'])
        mini_batch = np.array(mini_batch).transpose()

        states = np.vstack(mini_batch[0])
        actions = list(mini_batch[1])
        rewards = list(mini_batch[2])
        next_states = np.vstack(mini_batch[3])
        dones = mini_batch[4].astype(int)

        states = torch.Tensor(states).to(self.device)
        next_states = torch.Tensor(next_states).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        dones = torch.Tensor(dones).to(self.device)

        states = Variable(states).float()
        pred = self.model.forward(states, batch_size=self.args['batch_size']*self.args['skip'])

        # one-hot encoding
        a = torch.LongTensor(actions).view(-1, 1)

        one_hot_action = torch.FloatTensor(self.args['batch_size']*self.args['skip'], self.action_dim).zero_()
        one_hot_action.scatter_(1, a, 1)
        one_hot_action = torch.Tensor(one_hot_action).to(self.device)

        pred = torch.sum(pred.mul(Variable(one_hot_action)), dim=1)

        # Q Learning: get maximum Q value at s' from target model
        q_next_select = self.model(next_states, batch_size=self.args['batch_size']*self.args['skip'])
        _, q_next_select = torch.max(q_next_select, 1)

        one_hot_action = torch.FloatTensor(self.args['batch_size']*self.args['skip'], self.action_dim).zero_()
        one_hot_action.scatter_(1, a, 1)
        one_hot_action = torch.Tensor(one_hot_action).to(self.device)

        target_pred = self.target_model(next_states, batch_size=self.args['batch_size']*self.args['skip'])
        target_pred = torch.sum(target_pred.mul(Variable(one_hot_action)), dim=1)

        target = rewards + (1 - dones) * self.args['gamma'] * target_pred
        target = Variable(target)

        self.model_optimizer.zero_grad()

        # MSE Loss function
        loss = F.mse_loss(pred, target)
        loss.backward()

        # and train
        self.model_optimizer.step()