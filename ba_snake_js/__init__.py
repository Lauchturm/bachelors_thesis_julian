import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Planar-direction-parquet-v0',
    entry_point='ba_snake_js.envs.planar_snake_car:PlanarDirectionParquet',
    max_episode_steps=1024,
)

register(
    id='Planar-locomotion-parquet-v0',
    entry_point='ba_snake_js.envs.planar_snake_car:PlanarLocomotionParquet',
    max_episode_steps=1024,
)

register(
    id='Planar-script-enjoy-line-v0',
    entry_point='ba_snake_js.envs.planar_snake_car:PlanarScriptStraightEnjoyLine',
    max_episode_steps=1024,
)

register(
    id='Planar-direction-enjoy-line-slow-v0',
    entry_point='ba_snake_js.envs.planar_snake_car:PlanarDirectionSlowEnjoyLine',
    max_episode_steps=1024,
)

register(
    id='Planar-direction-enjoy-line-v0',
    entry_point='ba_snake_js.envs.planar_snake_car:PlanarDirectionEnjoyLine',
    max_episode_steps=1024,
)

register(
    id='Planar-locomotion-enjoy-line-v0',
    entry_point='ba_snake_js.envs.planar_snake_car:PlanarLocomotionEnjoyLine',
    max_episode_steps=1024,
)

register(
    id='Planar-obstacle-enjoy-five-v1',
    entry_point='ba_snake_js.envs.planar_snake_car:PlanarObstacleEnjoyFiveHighRes',
    max_episode_steps=1024,
)

register(
    id='Planar-obstacle-enjoy-five-v0',
    entry_point='ba_snake_js.envs.planar_snake_car:PlanarObstacleEnjoyFive',
    max_episode_steps=1024,
)

register(
    id='Planar-obstacle-v2',
    entry_point='ba_snake_js.envs.planar_snake_car:PlanarObstacle3',
    max_episode_steps=1024,
)

register(
    id='Planar-obstacle-v1',
    entry_point='ba_snake_js.envs.planar_snake_car:PlanarObstacle2',
    max_episode_steps=1024,
)

register(
    id='Planar-obstacle-v0',
    entry_point='ba_snake_js.envs.planar_snake_car:PlanarObstacle',
    max_episode_steps=1024,
)

register(
    id='Planar-locomotion-enjoy-circle-v0',
    entry_point='ba_snake_js.envs.planar_snake_car:PlanarLocomotionEnjoyCircle',
    max_episode_steps=1024,
)

register(
    id='Planar-locomotion-v1',
    entry_point='ba_snake_js.envs.planar_snake_car:PlanarLocomotion',
    max_episode_steps=1024,
)

register(
    id='Planar-direction-enjoy-circle-slow-v0',
    entry_point='ba_snake_js.envs.planar_snake_car:PlanarDirectionSlowEnjoyCircle',
    max_episode_steps=1024,
)

register(
    id='Planar-direction-enjoy-circle-v0',
    entry_point='ba_snake_js.envs.planar_snake_car:PlanarDirectionEnjoyCircle',
    max_episode_steps=1024,
)

register(
    id='Planar-direction-v1',
    entry_point='ba_snake_js.envs.planar_snake_car:PlanarDirectionFrequency2',
    max_episode_steps=1024,
)

register(
    id='Planar-direction-v0',
    entry_point='ba_snake_js.envs.planar_snake_car:PlanarDirection',
    max_episode_steps=1024,
)

# register(
#     id='Acmr-just-dir-v0',
#     entry_point='ba_snake_js.envs.acmr_01:NewAcmrContinuous',
#     max_episode_steps=1024,
# )
#
# register(
#     id='Acmr-obstacle-more-img-enjoy-five-v0',
#     entry_point='ba_snake_js.envs.acmr_01:AcmrObstacleMoreImgEnjoyFive',
#     max_episode_steps=2048,
# )
#
# register(
#     id='Acmr-obstacle-more-img-enjoy-in-front-v0',
#     entry_point='ba_snake_js.envs.acmr_01:AcmrObstacleMoreImgEnjoyInFront',
#     max_episode_steps=2048,
# )
#
# register(
#     id='Acmr-obstacle-more-img-v0',
#     entry_point='ba_snake_js.envs.acmr_01:AcmrObstacleMoreImg',
#     max_episode_steps=2048,
# )
#
# register(
#     id='Acmr-obstacle-v0',
#     entry_point='ba_snake_js.envs.acmr_01:AcmrObstacle',
#     max_episode_steps=1024,
# )
#
# register(
#     id='Three-d-locomote-v0',
#     entry_point='ba_snake_js.envs.three_d:ThreeDLocom',
#     max_episode_steps=1024,
# )
#
# register(
#     id='Acmr-locomote-v0',
#     entry_point='ba_snake_js.envs.acmr_01:NewAcmrLocom',
#     max_episode_steps=1024,
# )
#
# register(
#     id='Planar-locomotion-v0',
#     entry_point='ba_snake_js.envs.planar_snake_car:NewPlanarLocom',
#     max_episode_steps=1024,
# )

# register(
#     id='Planar-continuous-v1',
#     entry_point='ba_snake_js.envs.planar_snake_car:NewPlanarContinuous',
#     max_episode_steps=1024,
# )
#
# register(
#     id='Planar-continuous-v0',
#     entry_point='ba_snake_js.envs.planar_snake_car:PlanarContinuous',
#     max_episode_steps=1024,
# )
#
# register(
#     id='Planar-discrete-v1',
#     entry_point='ba_snake_js.envs.planar_snake_car:NewPlanarDiscrete',
#     max_episode_steps=1024,
# )
#
# register(
#     id='Planar-discrete-v0',
#     entry_point='ba_snake_js.envs.planar_snake_car:PlanarDiscrete',
#     max_episode_steps=1024,
# )
#
# register(
#     id='Acmr-discrete-v0',
#     entry_point='ba_snake_js.envs.acmr_01:AcmrDiscrete',
#     max_episode_steps=600,
# )
#
# register(
#     id='Planar-discrete-free-target-v0',
#     entry_point='ba_snake_js.envs.planar_snake_car:PlanarDiscreteFreeTarget',
#     max_episode_steps=1024,
# )

# register(
#     id='ErgoBall-v0',
#     entry_point='gym_vrep.envs:ErgoBallEnv',
#     timestep_limit=100,
#     reward_threshold=10.0,
#     nondeterministic=False,
# )