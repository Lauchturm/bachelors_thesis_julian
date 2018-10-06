# from baselines.ppo1 import pposgd_simple
from utils import my_pposgd_simple
from baselines import logger, bench
import gym
from baselines.common import set_global_seeds, tf_util as U
from baselines.ppo1 import mlp_policy, cnn_policy
import os
import tensorflow as tf

import ba_snake_js.envs.planar_snake_car  # TODO needed?

import random
from pathlib import Path
import definitions

# to check if unused number
logs_dir = Path(definitions.ROOT_DIR).joinpath('logs')


# just returns False for now as i prefer to stop it myself atm
def callback(lcl, _glb):
    # if len(lcl['rewbuffer']) > 99:  # lcl['timesteps_so_far'] > 500 and
    #     avg_last_fifty = sum(itertools.islice(lcl['rewbuffer'], 49, 99)) / 50
    #     is_solved = avg_last_fifty > 60
    #     if is_solved:
    #         print('is solved')
    #     else:
    #         print(f"50 durchschnitt: {avg_last_fifty}")
    #     return is_solved
    # else:
    #     return False
    return False  # prefer running until I'm content and then killing it in V-REP atm


def policy_fn_mlp(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                hid_size=64, num_hid_layers=2)


def policy_fn_cnn(name, ob_space, ac_space):
    return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space, kind='small')  # TODO kind='large'


def train_ppo1(env_name, num_timesteps, seed, model_path, logname, steps_per_batch):
    # log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'x_new')
    log_dir = str(Path.cwd().parent.parent.joinpath('logs', str(logname)))
    # win: tensorboard --logdir="E:\gitstuff\bachelor_thesis_snake\logs\x"
    logger.configure(dir=log_dir, format_strs=['stdout', 'log', 'csv', 'json', 'tensorboard'])
    # mac: go to cwd, pipenv shell, tensorboard --logdir=/Users/julian.schmitz/gitstuff/bachelor_thesis_snake/logs/1
    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    set_global_seeds(seed)
    env = gym.make(env_name)
    env = bench.Monitor(env, os.path.join(logger.get_dir(), "monitor.json"))

    # try block is to save model when timeout or stop pressed in vrep
    try:
        # params for most MLP policies:
        my_pposgd_simple.learn(
            env, policy_fn=policy_fn_mlp,  # policy_fn, TODO which works better?
            max_timesteps=num_timesteps,
            # max_episodes=1000,
            timesteps_per_actorbatch=steps_per_batch,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear', callback=callback, save_path=model_path
        )

        # params for run_atari (with cnn policy):
        # my_pposgd_simple.learn(env, policy_fn_cnn,
        #                        max_timesteps=int(num_timesteps * 1.1),
        #                        timesteps_per_actorbatch=2048,
        #                        clip_param=0.2, entcoeff=0.01,
        #                        optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64,
        #                        gamma=0.99, lam=0.95,
        #                        schedule='linear'
        #                        )

        # env.close()
        saver = tf.train.Saver()
        saver.save(sess, model_path)
        print('saved to ' + model_path)
    except RuntimeError:  # VREP died or I pressed stop there
        saver = tf.train.Saver()
        saver.save(sess, model_path)
        print('VREP died')
        print('saved to ' + model_path)
    # print("Saving model to planar_snake_continuous_model.pkl")
    # act.save("planar_snake_continuous_model.pkl")


# doesnt work as intended i think
def learn_on_ppo1(env_name, num_timesteps, seed, model_path, lognum, old_model_path=None):
    log_dir = str(Path.cwd().parent.parent.joinpath('logs', str(lognum)))
    # win: tensorboard --logdir="E:\gitstuff\bachelor_thesis_snake\logs\x"
    logger.configure(dir=log_dir, format_strs=['stdout', 'log', 'csv', 'json', 'tensorboard'])
    # mac: go to cwd, pipenv shell, tensorboard --logdir=/Users/julian.schmitz/gitstuff/bachelor_thesis_snake/logs/1
    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    set_global_seeds(seed)
    env = gym.make(env_name)
    env = bench.Monitor(env, os.path.join(logger.get_dir(), "monitor.json"))

    pi = None
    if old_model_path is not None:
        pi = cnn_policy.CnnPolicy('pi', env.observation_space, env.action_space)
        # tf.train.Saver().restore(sess, old_model_path)  # TODO does that load the model and we can go on?
        tf.train.Saver().restore(tf.get_default_session(),
                                 old_model_path)  # TODO does that load the model and we can learn on?

    my_pposgd_simple.learn(
        env, policy_fn=policy_fn_cnn,
        max_timesteps=num_timesteps,
        # max_episodes=1000,
        timesteps_per_actorbatch=2048,  # TODO or 4096 when obstacles
        clip_param=0.2, entcoeff=0.0,
        optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
        gamma=0.99, lam=0.95, schedule='linear', callback=callback, pi=pi
    )
    # env.close()
    saver = tf.train.Saver()
    saver.save(sess, model_path)
    print('saved to ' + model_path)
    # print("Saving model to planar_snake_continuous_model.pkl")
    # act.save("planar_snake_continuous_model.pkl")


def main(env_name="Acmr-obstacle-more-img-v0", steps_per_batch=2048):
    seeed = random.randint(1, 50000)  # TODO fine?
    print('seed: ' + seeed)
    num_bu = 1
    model_path = Path.cwd().joinpath('actors', env_name + '-' + str(num_bu), 'actor')
    log_name = env_name + '-' + str(num_bu)
    while logs_dir.joinpath(log_name).exists():
        num_bu += 1
        model_path = Path.cwd().joinpath('actors', env_name + '-' + str(num_bu), 'actor')
        log_name = env_name + '-' + str(num_bu)
    if not Path(model_path).parent.exists():
        Path(model_path).parent.mkdir()
    print('model path: ' + str(model_path))
    # num_timesteps is that high because i dont even want it to end that way but have to give a value
    train_ppo1(env_name=env_name, num_timesteps=10000000, seed=seeed, model_path=str(model_path), logname=log_name,
               steps_per_batch=steps_per_batch)


if __name__ == '__main__':
    # old or other experiments:
    # "Planar-continuous-v1"
    # "Planar-locomote-v0"

    # env_name = 'Planar-obstacle-v0'
    # env_name = 'Planar-obstacle-v1'
    # env_name = 'Planar-obstacle-v2'
    # env_name = 'Planar-locomotion-v1'
    # env_name = 'Planar-direction-v0'
    # env_name = 'Planar-direction-v1'
    env_name = 'Planar-locomotion-parquet-v0'

    # steps per batch: 2x max episode length
    # main(env_name=env_name, steps_per_batch=4096)  this is for obstacles because episodes are longer there
    main(env_name=env_name, steps_per_batch=2048)
