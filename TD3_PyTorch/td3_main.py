import gym
import torch
import tensorboardX
from td3 import TD3
import argparse
import os
import time


def main(args):
    env = gym.make(args['env_name'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    state_dim = env.observation_space.shape[0]


    td3 = TD3(args, action_dim, max_action, state_dim, device)
    summary = tensorboardX.SummaryWriter('./log/{}_td3_{}'.format(args['env_name'], args['noise_type']))

    timestep = 0
    start_time = time.time()
    for episode in range(args['max_episode']):
        episode_reward = 0
        state = env.reset()

        while True:
            action = td3.get_action(state)
            next_state, reward, done, info = env.step(action)
            td3.save(state, action, reward, next_state, int(done))
            episode_reward += reward
            state = next_state
            timestep += 1

            if td3.memory_counter > args['batch_size']: # BATCH_SIZE(64) 이상일 때 부터 train 시작
                td3.train()

            if done:
                print('episode: ', episode, '   reward : %.3f'%(episode_reward), '    timestep :', timestep)

                summary.add_scalar('reward/episode', episode_reward, episode)

                break

        if episode % args['save_freq'] == 0:
            if not os.path.exists('./SaveModel') :
                os.mkdir('./SaveModel')
            torch.save(td3.actor.state_dict(), './SaveModel/{}_td3_{}_{}'.format(args['env_name'], args['noise_type'], episode))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--env-name', default='LunarLanderContinuous-v2')
    parser.add_argument('--env-seed', default=0)
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--evaluate', default=False, type=bool)
    parser.add_argument('--model-directory', default='./SaveModel/Pendulum-v0_210', type=str)
    parser.add_argument('--max-episode', default=5000)
    parser.add_argument('--save-freq', default=100)


    parser.add_argument('--actor-lr', default=1e-3)
    parser.add_argument('--critic-lr', default=1e-3)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--noise-timestep', default=10000)
    parser.add_argument('--memory-size', default=200000)
    parser.add_argument('--noise_type', default='gaussian')
    parser.add_argument('--noise-delta', default=0.1)
    parser.add_argument('--polyak', default=0.01, help='soft target update param')
    parser.add_argument('--batch-size', default=100)
    parser.add_argument('--random-action-timestep', default=3000)
    parser.add_argument('--tau', default=5e-3)


    args = vars(parser.parse_args())

    main(args)