import gym
import torch
import tensorboardX
from agents import TD3
import argparse
import os
import numpy as np
import utils


def main(args):
    env = gym.make(args['env_name'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    state_dim = env.observation_space.shape[0]


    td3 = TD3(args, action_dim, max_action, state_dim, device)
    trained_actor = torch.load(args['model_directory'])
    td3.actor.load_state_dict(trained_actor)

    timestep = 0
    for episode in range(args['max_episode']):
        episode_reward = 0
        state = env.reset()
        state = utils.init_state(state)

        while True:
            action = td3.get_action(state)
            action = utils.carRace_output_to_action(action)
            tmp_reward = 0
            for i in range(4):
                tmp_next_state, reward, done, info = env.step(action)
                tmp_reward += reward
            env.render()
            tmp_next_state = utils._preprocess(tmp_next_state)
            tmp_next_state = tmp_next_state[np.newaxis, np.newaxis, :, :]
            state = np.append(tmp_next_state, state[:, :3, :, :], axis=1)
            episode_reward += tmp_reward
            timestep += 1

            if done:
                print('episode: ', episode, '   reward : %.3f'%(episode_reward), '    timestep :', timestep)
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=0)
    parser.add_argument('--env-name', default='CarRacing-v0')
    parser.add_argument('--env-seed', default=0)
    parser.add_argument('--render', default=True, type=bool)
    parser.add_argument('--evaluate', default=False, type=bool)
    parser.add_argument('--model-directory', default='./SaveModel/CarRacing-v0_td3_gaussian_600', type=str)
    parser.add_argument('--max-episode', default=5000)
    parser.add_argument('--save-freq', default=100)


    parser.add_argument('--actor-lr', default=0.001)
    parser.add_argument('--critic-lr', default=0.001)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--noise-timestep', default=10000)
    parser.add_argument('--memory-size', default=200000)
    parser.add_argument('--noise_type', default='none')
    parser.add_argument('--noise-delta', default=1.0)
    parser.add_argument('--batch-size', default=32)
    parser.add_argument('--random-action-timestep', default=3000)
    parser.add_argument('--tau', default=1e-3)


    args = vars(parser.parse_args())

    main(args)