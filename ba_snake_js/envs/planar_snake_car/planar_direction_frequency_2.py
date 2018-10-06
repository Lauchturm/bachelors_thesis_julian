from ba_snake_js.envs.planar_snake_car.planar_direction import PlanarDirection


class PlanarDirectionFrequency2(PlanarDirection):
    def __init__(self, server_addr='127.0.0.1', server_port=19997,
                 scene_name='2018-09-17-planar-direction-frequency-2.ttt'):
        super().__init__(server_addr=server_addr, server_port=server_port, scene_name=scene_name)
