import gym
import torch
import tensorboardX
import numpy as np
from agents import QRDQN
import argparse
import os
import utils


def main(args):
    env = gym.make(args['env_name'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]


    qrdqn = QRDQN(args, state_dim, action_dim, device)
    summary = tensorboardX.SummaryWriter('./log/{}_{}'.format(args['env_name'], 'qrdqn'))

    timestep = 0
    for episode in range(args['max_episode']):
        episode_reward = 0
        state = env.reset()

        while True:
            if args['random_action_timestep'] > timestep :
                select = env.action_space.sample()
            else :
                select = qrdqn.get_action(state)


            next_state, reward, done, info = env.step(select)

            qrdqn.save(state, select, reward, next_state, int(done))
            episode_reward += reward
            state = next_state
            timestep += 1

            if timestep % 100 == 0 :
                qrdqn.update_target()

            if timestep > args['replay_start_size']: # BATCH_SIZE(64) 이상일 때 부터 train 시작
                if timestep % args['skip'] == 0 :
                    qrdqn.train()


            if done:
                if episode % 1 == 0 :
                    print('episode: ', episode, '   reward : %.3f'%(episode_reward), '    timestep :', timestep, '  epsilon :', qrdqn.epsilon)

                summary.add_scalar('reward/timestep', episode_reward, timestep)
                break

        if episode % args['save_freq'] == 0:
            if not os.path.exists('./SaveModel') :
                os.mkdir('./SaveModel')
            torch.save(qrdqn.model.state_dict(), './SaveModel/{}_{}_{}'.format(args['env_name'], 'qrdqn', episode))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--env-name', default='CartPole-v0')
    parser.add_argument('--env-seed', default=0)
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--evaluate', default=False, type=bool)
    parser.add_argument('--model-directory', default='./SaveModel/Pendulum-v0_210', type=str)
    parser.add_argument('--max-episode', default=10000000)
    parser.add_argument('--save-freq', default=1000)


    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--num_quant', default=4)
    parser.add_argument('--memory-size', default=350000)
    parser.add_argument('--batch-size', default=32)
    parser.add_argument('--random-action-timestep', default=10000)
    parser.add_argument('--skip', default=4)
    parser.add_argument('--replay-start-size', default=15000)


    args = vars(parser.parse_args())

    main(args)