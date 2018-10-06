from ba_snake_js.envs.planar_snake_car import PlanarLocomotion
from ba_snake_js.envs.planar_snake_car import PlanarDirection


class PlanarLocomotionEnjoyCircle(PlanarLocomotion):

    def _get_episode_done(self):
        return super()._get_episode_done() or self.target_pos[0] < -55  # for same diagram

    def _calc_new_target_pos(self, sinus=True, circle=False, line=False):
        return super()._calc_new_target_pos(sinus=False, circle=True)


class PlanarDirectionEnjoyCircle(PlanarDirection):

    def _get_episode_done(self):
        return super()._get_episode_done() or self.target_pos[0] < -55  # for same diagram

    def _calc_new_target_pos(self, sinus=True, circle=False, line=False):
        return super()._calc_new_target_pos(sinus=False, circle=True)


class PlanarDirectionSlowEnjoyCircle(PlanarDirection):

    def __init__(self):
        super().__init__(scene_name='2018-08-10-planar-direciton-same-start-pos-as-locomotion-slow.ttt')

    def _get_episode_done(self):
        return super()._get_episode_done() or self.target_pos[0] < -55  # for same diagram

    def _calc_new_target_pos(self, sinus=True, circle=False, line=False):
        return super()._calc_new_target_pos(sinus=False, circle=True)
