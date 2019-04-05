import gym
import torch
import tensorboardX
import random
import numpy as np
from agents import Double_DQN_Cnn
import argparse
import os
from utils import preprocess, init_state


def main(args):
    env = gym.make(args['env_name'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]


    dqn = Double_DQN_Cnn(args, state_dim, action_dim, device)
    summary = tensorboardX.SummaryWriter('./log/{}_{}'.format(args['env_name'], 'double_dqn'))

    timestep = 0
    for episode in range(args['max_episode']):
        episode_reward = 0
        state = env.reset()
        state = init_state(state)

        while True:
            if args['random_action_timestep'] > timestep :
                select = env.action_space.sample()
            else :
                select = dqn.get_action(state)

            tmp_state = state.copy()
            for i in range(4) :
                next_state, reward, done, info = env.step(select)
                if i == 3 : break
                next_state = preprocess(tmp_state, next_state)
                tmp_state = next_state
            # env.render()
            next_state = preprocess(tmp_state, next_state)
            dqn.save(state, select, reward, next_state, int(done))
            episode_reward += reward
            state = next_state
            timestep += 1

            if timestep % 10 == 0 :
                dqn.update_target()

            if timestep > args['replay_start_size']: # BATCH_SIZE(64) 이상일 때 부터 train 시작
                if timestep % args['skip'] == 0 :
                    dqn.train()


            if done:
                if episode % 1 == 0 :
                    print('episode: ', episode, '   reward : %.3f'%(episode_reward), '    timestep :', timestep, '  epsilon :', dqn.epsilon)

                summary.add_scalar('reward/timestep', episode_reward, timestep)
                break

        if episode % args['save_freq'] == 0:
            if not os.path.exists('./SaveModel') :
                os.mkdir('./SaveModel')
            torch.save(dqn.model.state_dict(), './SaveModel/{}_{}_{}'.format(args['env_name'], 'dqn', episode))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--env-name', default='BreakoutDeterministic-v4')
    parser.add_argument('--env-seed', default=0)
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--evaluate', default=False, type=bool)
    parser.add_argument('--model-directory', default='./SaveModel/Pendulum-v0_210', type=str)
    parser.add_argument('--max-episode', default=10000000)
    parser.add_argument('--save-freq', default=1000)


    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--memory-size', default=800000)
    parser.add_argument('--batch-size', default=16)
    parser.add_argument('--random-action-timestep', default=30000)
    parser.add_argument('--skip', default=4)
    parser.add_argument('--replay-start-size', default=50000)


    args = vars(parser.parse_args())

    main(args)