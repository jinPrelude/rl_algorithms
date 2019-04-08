import gym
import torch
import tensorboardX
from agents import TD3
import argparse
import os


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

        while True:
            action = td3.get_action(state)
            next_state, reward, done, info = env.step(action)
            env.render()
            episode_reward += reward
            state = next_state
            timestep += 1

            if done:
                print('episode: ', episode, '   reward : %.3f'%(episode_reward), '    timestep :', timestep)
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--env-name', default='LunarLanderContinuous-v2')
    parser.add_argument('--env-seed', default=0)
    parser.add_argument('--render', default=True, type=bool)
    parser.add_argument('--evaluate', default=False, type=bool)
    parser.add_argument('--model-directory', default='./SaveModel/LunarLanderContinuous-v2_gaussian_100', type=str)
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