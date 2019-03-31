import gym
import torch
import random
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from memory import Memory, Per

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
            nn.Linear(state_dim+action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

    def forward(self, x, action, dim = 1):
        x = torch.cat([x, action], dim=dim)
        x = self.fc1(x)
        return x


class DDPG(nn.Module):
    def __init__(self, args, action_dim, max_action, state_dim, device):
        super(DDPG, self).__init__()
        self.actor = Actor(state_dim, action_dim).to(device)
        self.target_actor = Actor(state_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args['actor_lr'])

        self.critic = Critic(state_dim, action_dim).to(device)
        self.target_critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args['critic_lr'])

        self.memory = Memory(args['memory_size'])

        self.ou = OrnsteinUhlenbeckActionNoise(action_dim=action_dim)

        self.args = args
        self.action_dim = action_dim
        self.max_action = max_action
        self.state_dim = state_dim
        self.device = device

    def noise(self):
        return np.random.normal(0, self.args['noise_stddev'], self.action_dim)

    def get_action(self, state):
        state = torch.Tensor(state).to(self.device)
        model_action = self.actor(state).cpu().detach().numpy()
        if self.args['noise_type'] == 'ou' :
            action = model_action + self.ou.sample()
        elif self.args['noise_type'] == 'gaussian' :
            action = model_action + self.noise()
        elif self.args['noise_type'] == 'none' :
            action = model_action
        return action*self.max_action

    def update_target_model(self):
        target = [self.target_actor, self.target_critic]
        source = [self.actor,self.critic]
        for t,s in zip(target,source) :
            for target_param, param in zip(t.parameters(), s.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.args['tau']) + param.data * self.args['tau'])

    def save(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))


    def train(self):
        minibatch = self.memory.sample(self.args['batch_size'])
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

        # critic update
        self.critic_optimizer.zero_grad()
        value = self.critic(states, actions)
        target_value = self.target_critic(next_states, self.target_actor(next_states))
        target = rewards + (1 - dones) * self.args['gamma'] * target_value
        critic_loss = F.mse_loss(value, target)
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor update
        self.actor_optimizer.zero_grad()
        actor_loss = -self.critic(states, self.actor(states)).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_target_model()


class DDPG_PER(nn.Module):
    def __init__(self, args, action_dim, max_action, state_dim, device):
        super(DDPG_PER, self).__init__()
        self.actor = Actor(state_dim, action_dim).to(device)
        self.target_actor = Actor(state_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args['actor_lr'])

        self.critic = Critic(state_dim, action_dim).to(device)
        self.target_critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args['critic_lr'])

        self.memory = Memory(args['memory_size'])

        self.ou = OrnsteinUhlenbeckActionNoise(action_dim=action_dim)

        self.args = args
        self.action_dim = action_dim
        self.max_action = max_action
        self.state_dim = state_dim
        self.device = device

    def noise(self):
        return np.random.normal(0, self.args['noise_stddev'], self.action_dim)

    def get_action(self, state):
        state = torch.Tensor(state).to(self.device)
        model_action = self.actor(state).cpu().detach().numpy()
        if self.args['noise_type'] == 'ou' :
            action = model_action + self.ou.sample()
        elif self.args['noise_type'] == 'gaussian' :
            action = model_action + self.noise()
        elif self.args['noise_type'] == 'none' :
            action = model_action
        return action*self.max_action

    def update_target_model(self):
        target = [self.target_actor, self.target_critic]
        source = [self.actor,self.critic]
        for t,s in zip(target,source) :
            for target_param, param in zip(t.parameters(), s.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.args['tau']) + param.data * self.args['tau'])

    def save(self, state, action, reward, next_state, done):
        t_state = torch.Tensor(state).to(self.device)
        t_next_state = torch.Tensor(next_state).to(self.device)
        t_action = torch.Tensor(action).to(self.device)
        old_val = self.critic(t_state, t_action, dim=0)
        target_val = reward + (1-done) * self.args['gamma'] * self.target_critic(t_next_state, t_action, dim=0)
        error = abs(old_val - target_val)
        self.memory.add(error, (state, action, reward, next_state, done))


    def train(self):
        minibatch, idxs, is_weights = self.memory.sample(self.args['batch_size'])
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

        # critic update
        self.critic_optimizer.zero_grad()
        value = self.critic(states, actions)
        target_value = self.target_critic(next_states, self.target_actor(next_states))
        target = rewards + (1 - dones) * self.args['gamma'] * target_value
        critic_loss = F.mse_loss(value, target)
        critic_loss.backward()
        self.critic_optimizer.step()

        errors = torch.abs(value - target)
        errors = torch.squeeze(errors).cpu().detach().numpy()
        for i in range(self.args['batch_size']):
            idx = idxs[i]
            self.memory.update(idx, errors[i])

        # actor update
        self.actor_optimizer.zero_grad()
        actor_loss = -self.critic(states, self.actor(states)).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_target_model()
