from ba_snake_js.envs.planar_snake_car import PlanarLocomotion
from ba_snake_js.envs.planar_snake_car import PlanarDirection


class PlanarLocomotionEnjoyLine(PlanarLocomotion):

    def _calc_new_target_pos(self, sinus=True, circle=False, line=False):
        return super()._calc_new_target_pos(sinus=False, circle=False, line=True)

    def _get_episode_done(self):
        return super()._get_episode_done() or self.target_pos[0] > -10  # for same diagram


class PlanarDirectionEnjoyLine(PlanarDirection):

    def _calc_new_target_pos(self, sinus=True, circle=False, line=False):
        return super()._calc_new_target_pos(sinus=False, circle=False, line=True)

    def _get_episode_done(self):
        return super()._get_episode_done() or self.target_pos[0] > -10  # for same diagram


class PlanarDirectionSlowEnjoyLine(PlanarDirection):

    def __init__(self):
        super().__init__(scene_name='2018-08-10-planar-direciton-same-start-pos-as-locomotion-slow.ttt')

    def _calc_new_target_pos(self, sinus=True, circle=False, line=False):
        return super()._calc_new_target_pos(sinus=False, circle=False, line=True)

    def _get_episode_done(self):
        return super()._get_episode_done() or self.target_pos[0] > -10  # for same diagram


class PlanarScriptStraightEnjoyLine(PlanarDirection):

    def __init__(self, server_addr='127.0.0.1', server_port=19997,
                 scene_name='2018-09-12-planar-direction-script-straight.ttt'):
        super().__init__(server_addr=server_addr, server_port=server_port, scene_name=scene_name)

    def _calc_new_target_pos(self, sinus=True, circle=False, line=False):
        return super()._calc_new_target_pos(sinus=False, circle=False, line=True)

    def _get_episode_done(self):
        return super()._get_episode_done() or self.target_pos[0] > -10  # for same diagram
