import random

from baselines import logger, bench
import gym
from baselines.common import set_global_seeds, tf_util as U
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir)

import ba_snake_js.envs.planar_snake_car


from ppo1 import policy_fn_mlp


def enjoy(env_name, seed, model_path, num_enjoys=1, plot=0):
    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    set_global_seeds(seed)
    env = gym.make(env_name)
    # env = bench.Monitor(env, os.path.join(logger.get_dir(), "monitor.json"))
    pi = policy_fn_mlp('pi', env.observation_space, env.action_space)
    tf.train.Saver().restore(sess, model_path)
    infos = {
        'head_pos': {
            'x': [],
            'y': [],
        },
        'target_pos': {
            'x': [],
            'y': [],
        },
        'num_lost_target': 0,
        'num_episodes': num_enjoys,
    }
    for i in range(num_enjoys):
        obs = env.reset()
        done = False
        # while not done:
        for j in range(300):
            action = pi.act(True, obs)[0]
            obs, reward, done, info = env.step(action)
            # info for diagrams like head positions
            infos['head_pos']['x'].append(info['head_pos']['x'])
            infos['head_pos']['y'].append(info['head_pos']['y'])
            infos['target_pos']['x'].append(info['target_pos']['x'])
            infos['target_pos']['y'].append(info['target_pos']['y'])
            infos['num_lost_target'] += info['num_lost_target']
            # print(info)
            env.render()

        if plot == 1:
            plot_trajectory(infos, i)
        elif plot == 2:
            plot_histogram(infos, i)
        elif plot == 3:
            plot_joint_positions(infos, i)

        infos = {
            'head_pos': {
                'x': [],
                'y': [],
            },
            'target_pos': {
                'x': [],
                'y': [],
            },
            'num_lost_target': 0,
            'num_episodes': num_enjoys,
        }  # TODO because needed granular plots
        env.reset()
    return infos


def main(actor_name, env_name, num_enjoys=1, plot=0):
    seeed = random.randint(1, 50000)  # TODO take the same for eval?
    model_path = os.path.join(os.getcwd(), 'actors', str(actor_name), 'actor')
    if 'iter' in model_path:
        model_path = os.path.join(os.getcwd(), 'actors', str(actor_name), 'model')
    infos = enjoy(env_name=env_name, seed=seeed, model_path=model_path, num_enjoys=num_enjoys, plot=plot)
    # print(infos)
    # for i in range(num_enjoys):
    #     if plot == 1:
    #         plot_trajectory(infos, i)
    #     elif plot == 2:
    #         plot_histogram(infos, i)


def plot_trajectory(infos, i):
    head_pos = infos['head_pos']
    target_pos = infos['target_pos']
    num_lost_target = infos['num_lost_target']

    # print(f'lost: {num_lost_target} times of {infos["num_episodes"]}')
    print('lost: {} times of {}'.format(num_lost_target, infos["num_episodes"]))

    # plt.axis('equal')
    # plt.plot(head_pos['x'], head_pos['y'], color='green')
    # plt.plot(target_pos['x'], target_pos['y'], color='red')
    # plt.show()

    df = pd.DataFrame(
        {'head_x': head_pos['x'],
         'head_y': head_pos['y'],
         'target_x': target_pos['x'],
         'target_y': target_pos['y'],
         }
    )

    # df.to_csv(f'positions_{i}.csv')  # save it for later
    df.to_csv('positions_{}.csv'.format(i))  # save it for later

    # df = pd.DataFrame(
    #     {'x': head_pos['x'],
    #      'y': head_pos['y'],
    #      'Object': 0,
    #      })
    # df2 = pd.DataFrame.from_dict(
    #     {
    #         'x': target_pos['x'],
    #         'y': target_pos['y'],
    #         'Object': 1,
    #     }
    # )
    # print(df)
    # print(df2)
    # df = df.append(df2)  # , ignore_index=True)
    # print(df)

    sns.set()
    sns.set_context("talk")
    sns.axes_style('ticks')
    # ax = sns.relplot(hue='Object', data=df)
    # ax.axis('equal')
    ax = sns.lineplot(x=df['head_x'], y=df['head_y'], sort=False, label='Snake')
    ax2 = sns.lineplot(x=df['target_x'], y=df['target_y'], sort=False, label='Target')
    ax.axis('equal')
    ax2.axis('equal')
    ax.set(xlabel='x', ylabel='y')
    ax.legend()
    # sns.despine()
    plt.show()
    fig = ax.get_figure()
    fig.savefig('myplot_{}.pdf'.format(i), dpi=300, bbox_inches='tight')
    fig.savefig('myplot_{}.png'.format(i), dpi=300, bbox_inches='tight')
    # df2 = pd.DataFrame(
    #     {
    #     # {'head_x': head_pos['x'],
    #     #  'head_y': head_pos['y'],
    #      'target_x': target_pos['x'],
    #      'target_y': target_pos['y'],
    #      })
    # print(df)
    # df.plot()

    # fig, ax = plt.subplots()
    # ax2 = ax.twinx()
    # sns.regplot(x="head_x", y="head_y", data=df, order=2, ax=ax)
    # sns.regplot(x="target_x", y="target_y", data=df, order=2, ax=ax2)

    # sns.relplot(data=df)
    # sns.relplot(data=df2)
    # plt.show()


def plot_histogram(infos, i):
    head_pos = infos['head_pos']

    df = pd.DataFrame(
        {'head_x': head_pos['x'],
         'head_y': head_pos['y'],
         }
    )

    df.to_csv('positions_{}.csv'.format(i))  # save it for later

    sns.set()
    sns.set_context("talk")

    ax = sns.distplot(df['head_y'], hist=True, rug=True)
    plt.show()


def plot_joint_positions(infos, i):
    pass


if __name__ == '__main__':
    sns.set()
    sns.set_context("talk")

    # actor_name = 'Planar-locomotion-v1-2'
    # actor_name = 'Planar-direction-v0-3'
    # actor_name = 'Planar-obstacle-v0-1'  # TODO not the right actor? just running circles
    # actor_name = 'Planar-obstacle-v2-1/iter_300'
    # env_name = 'Planar-obstacle-enjoy-five-v0'
    # env_name = 'Planar-obstacle-enjoy-five-v1'
    # actor_name = 'Planar-direction-v0-3'
    # actor_name = 'Planar-locomotion-v1-2'
    actor_name = 'Planar-direction-v1-2'
    # env_name = 'Planar-locomotion-v1'
    # env_name = 'Planar-locomotion-enjoy-circle-v0'
    # env_name = 'Planar-direction-enjoy-circle-v0'
    # env_name = 'Planar-direction-enjoy-circle-slow-v0'
    # env_name = 'Planar-direction-enjoy-line-v0'
    # env_name = 'Planar-direction-enjoy-line-slow-v0'
    # env_name = 'Planar-locomotion-enjoy-line-v0'
    # env_name = 'Planar-script-enjoy-line-v0'
    # env_name = 'Planar-direction-v0'

    # parquet envs
    env_name = 'Planar-direction-parquet-v0'
    # env_name = 'Planar-locomotion-parquet-v0'

    # training envs
    # env_name = 'Planar-direction-v1'
    # env_name = 'Planar-locomotion-v1'

    num_enjoys = 1
    main(actor_name=actor_name, env_name=env_name, num_enjoys=num_enjoys, plot=0)
