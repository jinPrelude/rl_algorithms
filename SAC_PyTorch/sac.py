import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class OrnsteinUhlenbeckActionNoise(object):
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.fc1(x)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 200)
        )

    def forward(self, x, action):
        x = torch.cat([x, action], dim=1)
        x = self.fc1(x)
        return x

class Value(nn.Module):
    def __init__(self, state_dim):
        super(Value, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

    def forward(self, x):
        x = self.fc1(x)
        return x


class SAC(nn.Module):
    def __init__(self, args, action_dim, max_action, state_dim, device):
        super(SAC, self).__init__()
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args['actor_lr'])

        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer1 = optim.Adam(self.critic1.parameters(), lr=args['critic_lr'])
        self.critic_optimizer2 = optim.Adam(self.critic2.parameters(), lr=args['critic_lr'])

        self.value = Value(state_dim).to(device)
        self.target_value = Value(state_dim).to(device)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=args['critic_lr'])

        self.memory = deque(maxlen=args['memory_size'])
        self.memory_counter = 0

        self.ou = OrnsteinUhlenbeckActionNoise(action_dim=action_dim)

        self.args = args
        self.action_dim = action_dim
        self.max_action = max_action
        self.state_dim = state_dim
        self.device = device

    def noise(self, delta=0.1):
        return np.random.normal(0, delta, self.action_dim)

    def get_action(self, state):
        state = torch.Tensor(state).to(self.device)
        model_action = self.actor(state).cpu().detach().numpy()
        if self.args['noise_type'] == 'ou':
            action = model_action + self.ou.sample()
        elif self.args['noise_type'] == 'gaussian':
            action = model_action + self.noise(self.args['noise_delta'])
        elif self.args['noise_type'] == 'none':
            action = model_action
        return action * self.max_action

    def update_target_model(self):
        target = [self.value]
        source = [self.target_value]
        for t, s in zip(target, source):
            for target_param, param in zip(t.parameters(), s.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.args['tau']) + param.data * self.args['tau'])

    def save(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.memory_counter += 1

    def train(self):
        minibatch = random.sample(self.memory, self.args['batch_size'])
        minibatch = np.array(minibatch).transpose()

        states = torch.from_numpy(np.vstack(minibatch[0])).float()
        actions = torch.from_numpy(np.vstack(minibatch[1])).float()
        rewards = torch.from_numpy(np.vstack(minibatch[2])).float()
        next_states = torch.from_numpy(np.vstack(minibatch[3])).float()
        dones = torch.from_numpy(np.vstack(minibatch[4].astype(int))).float()

        states = torch.Tensor(states).to(self.device)
        actions = torch.Tensor(actions).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        next_states = torch.Tensor(next_states).to(self.device)
        dones = torch.Tensor(dones).to(self.device)

        # reward normalization
        #rewards_mean = torch.mean(rewards)
        #rewards_var = rewards.var()
        #scaled_reward = (rewards - rewards_mean) / rewards_var
        scaled_reward = rewards

        # critic update
        next_target_action = self.actor(states)
        q = torch.min(self.critic1(states, next_target_action), self.critic2(states, next_target_action))
        target_v = q - torch.log(next_target_action)
        target_q = (rewards + self.args['gamma']*(1-dones)*self.target_value(next_states)).detach()

        q1 = self.critic1(states, actions)
        critic_loss1 = F.mse_loss(q1, target_q)
        self.critic_optimizer1.zero_grad()
        critic_loss1.backward()
        self.critic_optimizer1.step()

        q2 = self.critic2(states, actions)
        critic_loss2 = F.mse_loss(q2, target_q)
        self.critic_optimizer2.zero_grad()
        critic_loss2.backward()
        self.critic_optimizer2.step()

        value_loss = F.mse_loss(self.value(states), target_v)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # actor update
        self.actor_optimizer.zero_grad()
        entropy = torch.Tensor(self.args['alpha']*torch.log(self.actor(states))).to(self.device)
        actor_loss = -(self.critic1(states, self.actor(states))-entropy).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_target_model()


