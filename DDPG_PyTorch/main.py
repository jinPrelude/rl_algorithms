import gym
import torch
import tensorboardX
from ddpg import DDPG
import argparse
import os
import time


def main(args):
    env = gym.make(args['env_name'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    state_dim = env.observation_space.shape[0]


    ddpg = DDPG(args, action_dim, max_action, state_dim, device)
    summary = tensorboardX.SummaryWriter('./log/{}_{}'.format(args['env_name'], args['noise_type']))

    timestep = 0
    start_time = time.time()
    for episode in range(args['max_episode']):
        episode_reward = 0
        state = env.reset()

        while True:
            action = ddpg.get_action(state)
            next_state, reward, done, info = env.step(action)
            ddpg.save(state, action, reward, next_state, int(done))
            episode_reward += reward
            state = next_state
            timestep += 1

            if ddpg.memory_counter > args['batch_size']: # BATCH_SIZE(64) 이상일 때 부터 train 시작
                ddpg.train()

            if done:
                print('episode: ', episode, '   reward : %.3f'%(episode_reward), '    timestep :', timestep)

                summary.add_scalar('reward/episode', episode_reward, episode)
                sec = int(start_time - time.time())
                summary.add_scalar('reward/time', episode_reward, sec)

                break

        if episode % args['save_freq'] == 0:
            if not os.path.exists('./SaveModel') :
                os.mkdir('./SaveModel')
            torch.save(ddpg.actor.state_dict(), './SaveModel/{}_{}_{}'.format(args['env_name'], args['noise_type'], episode))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--env-name', default='LunarLanderContinuous-v2')
    parser.add_argument('--env-seed', default=0)
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--evaluate', default=False, type=bool)
    parser.add_argument('--model-directory', default='./SaveModel/Pendulum-v0_210', type=str)
    parser.add_argument('--max-episode', default=5000)
    parser.add_argument('--save-freq', default=100)


    parser.add_argument('--actor-lr', default=0.001)
    parser.add_argument('--critic-lr', default=0.001)
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