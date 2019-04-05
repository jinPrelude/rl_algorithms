import gym
import torch
import random
import numpy as np
from agents import Double_DQN_Cnn
import argparse
from utils import init_state, preprocess
import os
import time

def main(args):
    env = gym.make(args['env_name'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]


    dqn = Double_DQN_Cnn(args, state_dim, action_dim, device)
    dqn.model.load_state_dict(torch.load('./SaveModel/BreakoutDeterministic-v4_dqn_3200'))

    while True :
        state = env.reset()
        state = init_state(state)
        while True:
            select = dqn.get_real_action(state)
            next_state, reward, done, info = env.step(select)
            env.render()
            time.sleep(0.03)

            next_state = preprocess(state, next_state)
            state = next_state

            if done:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--env-name', default='BreakoutDeterministic-v4')
    parser.add_argument('--env-seed', default=0)
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--evaluate', default=False, type=bool)
    parser.add_argument('--model-directory', default='./SaveModel/Pendulum-v0_210', type=str)
    parser.add_argument('--max-episode', default=5000)
    parser.add_argument('--save-freq', default=100)


    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--noise-timestep', default=10000)
    parser.add_argument('--memory-size', default=200000)
    parser.add_argument('--noise_type', default='gaussian')
    parser.add_argument('--noise-stddev', default=1.0)
    parser.add_argument('--polyak', default=0.01, help='soft target update param')
    parser.add_argument('--batch-size', default=32)
    parser.add_argument('--random-action-timestep', default=3000)
    parser.add_argument('--tau', default=1e-3)


    args = vars(parser.parse_args())

    main(args)