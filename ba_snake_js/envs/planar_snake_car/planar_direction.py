from ba_snake_js.envs.planar_snake_car.planar_snake_car_base import PlanarBase
from gym import spaces
import numpy as np
from utils.vrep_env import vrep


class PlanarDirection(PlanarBase):
    def __init__(self, server_addr='127.0.0.1', server_port=19997,
                 scene_name='2018-08-10-planar-direction-same-start-pos-as-locomotion.ttt'):
        super().__init__(server_addr=server_addr, server_port=server_port, scene_name=scene_name)
        self.action_space = spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)

    def _make_action(self, action):
        """Query V-REP to make action
            no return value
        """
        _, self.joint_positions, _, _ = self.RAPI_rc(
            vrep.simxCallScriptFunction(self.cID, self.name_snake, 1, 'setDirection_cb', [], [action], [],
                                        bytearray(), vrep.simx_opmode_blocking))
        # print(f'res: {self.joint_positions}')
