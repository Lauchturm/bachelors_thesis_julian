# Comparison of Reinforcement Learning for Direct and Indirect Locomotion Control in Target Tracking with Snake-like Robots
# Bachelor's Thesis<br/>Julian Schmitz<br/>Technical University Munich

Two approaches for a Snake-like robot tracking a target using V-REP and Python.

vrep.py, vrepConst.py, remoteApi.dll and remoteApi.dylib are in this dir for ease of use and ease of adjustments.

Installation
- V-REP (I used 3.5 EDU): follow instructions on http://www.coppeliarobotics.com/
- install python3
- install python3 modules: `pip install -r /path/to/requirements.txt`
- tensorflow: follow the instructions on https://www.tensorflow.org/install/
- baselines: follow the instructions on https://github.com/openai/baselines
- set env vars according to instructions on https://github.com/ycps/vrep-env
- you may have to install python3-tk if the error of missing tkinter arises

## To run a learned agent (the files in "actors"):

pick one of the enjoy files in agents e.g. enjoy_direct_eval.py

start v-rep

`python3 enjoy_direct_eval.py`

## To view the training details (logs), (install and) run Tensorboard:

on win: 

`tensorboard --logdir="path\to\repo\logs\logname"`

mac, linux first have to enter the python interactive shell and then run tensorboard: 

`python3 `

`tensorboard --logdir=path/to/repo/logs/logname`

## Utils

Contains V-REP RemoteAPI for sake of easy deployment, adapted versions of vrep_env for mac sleep fixes, adapted version
of baselines pposgd_simple to make saving easier.

## Plots & Figures

data is mostly obtained in ppo1_enjoy.py and plotted in plot_csvs.py

xml files are created with draw.io