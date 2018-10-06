from utils.vrep_env import vrep_env, vrep

import os

from gym.utils import seeding
from gym import spaces
import numpy as np

import math
from definitions import ROOT_DIR, sinus_y_step, x_delta_target
import random
import time
import logging

# vrep_scenes_path = os.environ['VREP_SCENES_PATH']  # other way: via environment var
vrep_scenes_path = os.path.join(ROOT_DIR, 'ba_snake_js', 'scenes')
verbose = False


class PlanarBase(vrep_env.VrepEnv):
    def __init__(self, server_addr='127.0.0.1', server_port=19997, scene_name='planar_static_ground_bigger.ttt',
                 cam_res=32, num_joints=8, snake_name='PlanarSnakeCar', module_1_name='Car_joint_1',
                 vision_name='vision_sensor', target_name='Target', observation_space=None, radius=40,
                 gather_info=True):  # TODO turn off when not gathering
        scene_path = os.path.join(vrep_scenes_path, scene_name)
        super().__init__(server_addr, server_port, scene_path)

        self.name_snake = snake_name
        self.snake = self.get_object_handle(self.name_snake)
        self.module_1 = self.get_object_handle(module_1_name)

        # for 3d experiment without cam and target
        if vision_name is not None:
            self.vision_sensor = self.get_object_handle(vision_name)
        if target_name is not None:
            self.target = self.get_object_handle(target_name)

        self.num_joints = num_joints
        self.cam_res = cam_res
        self.observation_len = num_joints + cam_res + 1  # 1 for the speed of the head module
        # print(f'obs len:{self.observation_len}')
        if observation_space is not None:
            self.observation_space = observation_space
        else:
            self.observation_space = spaces.Box(low=-10, high=10, shape=(self.observation_len,), dtype=np.float32)

        self.timesteps_into_sinus_way = 90  # only setting for the first move
        self.min_distance = 5
        self.circle_first = True
        # self.circle_stepsize = 0.003
        # self.circle_stepsize = 0.0015
        self.circle_stepsize = 0.003
        # self.circle_stepsize = 0.0012
        self.radius = radius
        self.initial_target_pos = self.obj_get_position(self.target, -1)
        self.out_of_trajectory_start = False

        self.timesteps = 0
        self.episodes = 0
        self.this_episode_reward = 0
        self.head_speed = 0

        # TODO working with dqn?
        if target_name is not None:
            self.target_pos = self.obj_get_position(self.target)
        self.observation = np.zeros(self.observation_space.shape)
        self.joint_positions = np.zeros(self.num_joints)
        self.snake_pos = self.obj_get_position(self.module_1)
        self.old_snake_pos = self.snake_pos

        # to gather information like head positions for diagrams
        self.gather_info = gather_info
        self.num_lost_target = 0
        logging.basicConfig(filename='rewardlog.csv', level=logging.DEBUG, format='%(message)s')

    def reset(self):
        if self.this_episode_reward != 0:
            self.episodes += 1
            print(f'episode {self.episodes} reward: {self.this_episode_reward}')
            logging.info(f'{self.episodes}, {self.this_episode_reward}')

        # TODO maybe reset like this again?
        # if not self.sim_running:
        #     self.start_simulation()
        # self._reset_positions()
        # TODO old way to reset:
        if self.sim_running:
            self.stop_simulation()
        self.start_simulation()

        self._reset_vars()

        # self.snake_pos = self.obj_get_position(self.module_1)
        # self.old_snake_pos = self.snake_pos
        # self.joint_positions = np.zeros(self.num_joints)

        self._make_observation()
        return self.observation

    def step(self, action):
        # Clip or Assert
        # actions = np.clip(actions,-self.joints_max_velocity, self.joints_max_velocity)
        # assert self.action_space.contains(action), "Action {} ({}) is invalid".format(action, type(action))
        if not self.action_space.contains(action):
            if verbose:
                print(f'action {action} out of action space')
            pass

        self._make_action(action)
        self.step_simulation()  # telling V-REP to procede to the next step
        self._make_observation()

        self._move_target()
        reward = self._get_reward()
        if verbose:
            print(f'reward: {reward}')

        self.timesteps += 1
        done = self._get_episode_done()
        if done:
            actual_episode = self.episodes + 1
            logging.info(f'{actual_episode},fail')
        info = {}
        if self.gather_info:
            info = {
                "head_pos": {
                    'x': self.snake_pos[0],
                    'y': self.snake_pos[1]
                },
                "target_pos": {
                    'x': self.target_pos[0],
                    'y': self.target_pos[1]
                },
                "num_lost_target": self.num_lost_target
                # "timesteps": self.timesteps,
                # TODO sth to see if episode stopped early
            }

        return self.observation, reward, done, info

    def render(self, mode='human', close=False):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)  # TODO self.np_random needed? should i use it?
        return [seed]

    def _reset_vars(self):
        # init of some helper vars

        self.snake_pos = self.obj_get_position(self.module_1)
        self.old_snake_pos = self.snake_pos
        self.joint_positions = np.zeros(self.num_joints)
        self.target_pos = self.obj_get_position(self.target)
        self.initial_target_pos = self.target_pos
        self.out_of_trajectory_start = False
        self.old_target_pos = self.target_pos

        self.circle_phi = 0
        self.this_episode_reward = 0
        self.timesteps = 0
        self.up = True
        self.head_speed = 0
        self.distance = self._euclid_dist(self.old_target_pos[0], self.old_target_pos[1], self.snake_pos[0],
                                          self.snake_pos[1])

        self.circle_first = True

    def _get_reward(self):
        self.distance = self._euclid_dist(self.old_target_pos[0], self.old_target_pos[1], self.snake_pos[0],
                                          self.snake_pos[1])
        distance_from_old_snake_pos = self._euclid_dist(self.old_target_pos[0], self.old_target_pos[1],
                                                        self.old_snake_pos[0], self.old_snake_pos[1])

        delta_distance = distance_from_old_snake_pos - self.distance
        self.this_episode_reward += delta_distance
        # print(f'reward so far: {self.this_episode_reward}')
        if verbose:
            print(f'old_snake pos: {self.old_snake_pos} ||||| new: {self.snake_pos}')
            print(
                f'my distance: {self.distance}\t\tdistance from old: {distance_from_old_snake_pos}\t\t'
                f'reward so far: {self.this_episode_reward}')
            # print(f'under sqrt: {under_sqrt}')
            # print(f'under OLD sqrt: {old_under_sqrt}')
            # print(self.target_pos[0], self.target_pos[1], self.snake_pos[0], self.snake_pos[1])
            # print(f'distance: {self.distance}')
            # print(f'distance_OLD: {distance_from_old_snake_pos}')
            # print(f'reward: {delta_distance}')
            print('-------------------------------------------------------------')
            time.sleep(0.5)
        return delta_distance

    def _move_target(self):
        # this one moves independant from snek
        self.old_snake_pos = self.snake_pos
        self.snake_pos = self.obj_get_position(self.module_1)
        self.target_pos = self.obj_get_position(self.target)  # only needs to be here when restarted
        new_pos = self._calc_new_target_pos()
        self.old_target_pos = self.target_pos
        self.obj_set_position(self.target, new_pos)
        self.target_pos = new_pos

    def _calc_new_target_pos(self, sinus=True, circle=False,
                             line=False):  # TODO better way for choosing different tracks?
        # for the first steps always go straight because thats better to learn (see citation in paper)
        # if self.timesteps < 150:
        if (not self.out_of_trajectory_start) and (self.target_pos[0] - self.initial_target_pos[0] < 8):
            # if self.timesteps == 0 and self.circle_first:
            #     # setting mid of circle
            #     print('set x start')
            #     self.x_start = self.target_pos[0]
            y_coord = self.target_pos[1]
            x_coord = max(self.target_pos[0] + x_delta_target, self.snake_pos[0] + self.min_distance)
            # print('old x pos')
            # print(self.target_pos[0])
            # print('new')
            # print(self.target_pos[0] + x_delta_target)
            new_pos = [x_coord, y_coord, self.target_pos[2]]
            return new_pos
        else:
            self.out_of_trajectory_start = True
            if sinus:
                if self.timesteps % self.timesteps_into_sinus_way == 0:
                    self.up = not self.up
                    self.timesteps_into_sinus_way = random.randint(30, 210)
                y_step = sinus_y_step if self.up else -sinus_y_step
                y_coord = self.target_pos[1] + y_step
                x_coord = max(self.target_pos[0] + x_delta_target, self.snake_pos[0] + self.min_distance)
                new_pos = [x_coord, y_coord, self.target_pos[2]]
                return new_pos
            if circle:
                # if self.timesteps <
                # init radius etc
                if self.circle_first:
                    self.circle_first = False
                    # self.circle_stepsize = 0.0014
                    # self.x_start = self.target_pos[0]  # x_start is center of circle
                    # self.x_start = -47  # x_start is center of circle
                    # target_moved = self.target_pos[0] - self.x_start  # -47 is start pos of target
                    # target_moved = self.target_pos[0] - (-47)  # -47 is start pos of target
                    # x_start = -40  # -47 is actual first x of target
                    # self.radius = target_moved  # this was too little and would faster snakes have easier angles
                    self.x_start = self.target_pos[0] - self.radius
                    # print('x_start:', self.x_start)
                    # print('radius:', self.radius)
                self.circle_phi += self.circle_stepsize
                new_x = self.radius * math.cos(self.circle_phi) + self.x_start
                new_y = - self.radius * math.sin(self.circle_phi) + 0  # y start is 0
                new_pos = [new_x, new_y, self.target_pos[2]]
                distance = self._euclid_dist(new_x, new_y, self.snake_pos[0], self.snake_pos[1])
                # make sure the distance is more than min_distance else go a bit further around the circle
                while distance < self.min_distance:
                    self.circle_phi += self.circle_stepsize
                    new_x = self.radius * math.cos(self.circle_phi) + self.x_start
                    new_y = - self.radius * math.sin(self.circle_phi) + 0  # y start is 0
                    new_pos = [new_x, new_y, self.target_pos[2]]
                    distance = self._euclid_dist(new_x, new_y, self.snake_pos[0], self.snake_pos[1])

                return new_pos
            if line:
                y_coord = self.target_pos[1]
                x_coord = max(self.target_pos[0] + x_delta_target, self.snake_pos[0] + self.min_distance)
                new_pos = [x_coord, y_coord, self.target_pos[2]]
                return new_pos

    @staticmethod
    def _euclid_dist(a_x, a_y, b_x, b_y):
        return math.sqrt((a_x - b_x) ** 2 + (a_y - b_y) ** 2)

    def _get_image(self):
        """Query V-REP to get vision sensor image.
            The observation is stored in self.observation
        """
        try:
            # i think the arrays go through line by line beginning at the top
            # self.observation = self._julians_obj_get_vision_image(self.vision_sensor, first=first)
            # vision_sensor_output = self._julians_obj_get_vision_image(self.vision_sensor, first=first)
            vision_sensor_output = self.obj_get_vision_image(self.vision_sensor)
            # print(vision_sensor_output)
            # TODO make this faster using np
            # for line in vision_sensor_output:

            new_line = []
            # target should be red pixels in this line, so it should suffice to only look here and use hot one encoding
            for square in vision_sensor_output[15]:
                if (square[0] >= 100) and (square[1] < 100) and (square[2] < 100):
                    new_line.append(1)
                else:
                    new_line.append(0)
            # self.observation = np.array(new_line)  # .astype('float32') why was that here? it's ints not floats
            observation = np.array(new_line)  # .astype('float32') why was that here? it's ints not floats
        except ValueError:
            # needed because throws Valueerror now and then?
            observation = np.zeros(self.observation_space.shape)  # if self.observation else ""
            print('obsfail')
        return observation

    def _make_observation(self):
        try:
            image = self._get_image()
            self._calc_head_speed()
            joints_and_speed = np.append(self.joint_positions, self.head_speed)
            self.observation = np.append(image, joints_and_speed)
            # print(self.observation)
        except ValueError:
            # needed because throws Valueerror now and then?
            self.observation = self.observation  # if self.observation else ""
            print('obsfail')

    def _calc_head_speed(self):
        # euclidean distance between old and new position divided by time -> head module speed
        delta = self._euclid_dist(self.old_snake_pos[0], self.old_snake_pos[1], self.snake_pos[0], self.snake_pos[1])
        self.head_speed = delta / 0.050  # 50ms is one timestep

    def _get_episode_done(self):
        if self.distance > 15:
            print(f'episode {self.episodes}: lost target. resetting')
            self.num_lost_target += 1
        return self.distance > 15

    # abstract methods
    def _make_action(self, action):
        raise NotImplementedError
