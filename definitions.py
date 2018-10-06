import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # TODO why does os.getcwd() not work?
sinus_y_step = 0.05  # to y per step in sinus
# acmr is slower lets try 0.02 plus the min distance thing
x_delta_target = 0.08  # 0.02 was too slow
# how far the target moves each timestep to the x direction in sinus movement 0.03 for script gait but locomotion learning was too fast
