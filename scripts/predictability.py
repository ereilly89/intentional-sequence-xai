#!/usr/bin/env python3
import argparse
from array2gif import write_gif
import gym
import gym_minigrid
from gym_minigrid.window import Window
from gym_minigrid.wrappers import *
import numpy
import time
import torch
from torch_ac.utils.penv import ParallelEnv
from typing import Counter
import utils


def redraw(img, isManual):
    if not args.agent_view:
        print("tile_size:"+str(args.tile_size))
        if isManual:
            img = env.render('rgb_array', tile_size=args.tile_size)
        else:
            img = env.envs[0].render('rgb_array', tile_size=args.tile_size)

    window.show_img(img)


def resetManual():
    if args.seed != -1:
        env.seed(args.seed)

    if int(args.episodes) > 0:
        valid = False
        while not valid:
            obs = env.reset(None)
            if env.hash() not in uniqueStartStates.keys():
                valid = True
                uniqueStartStates[env.hash()] = True
                print("unique: " + str(uniqueStartStates))
                args.episodes = str(int(args.episodes) - 1)
                state = saveState(env)
        redraw(obs, True)
        return state

    else:
        window.close()
        return None


def resetAI(stateInfo):
    if args.seed != -1:
        env.seed(args.seed)

    print("\n\nepisodes: " + str(args.episodes))
    if int(args.episodes) > 0:
        obs = env.envs[0].reset(None)
            
        # set the state of the environment
        env.envs[0].agent_pos = stateInfo["agent_pos"]
        env.envs[0].agent_dir = stateInfo["agent_dir"]

        redraw(obs, False)
        return obs
        
"""
def step(action, isManual=False):
    obs, reward, done, info = env.step(action)
    print('step=%s, reward=%.2f' % (env.step_count, reward))

    if done:
        print('done!')
        if isManual:
            resetManual()
        else:
            resetAI()
        #reset(isManual)
    else:
        redraw(obs, isManual)
"""

def stepManual(action):
    obs, reward, done, info = env.step(action)
    if done:
        stateInfo = resetManual()
        return stateInfo
    else:
        redraw(obs, True)
        return None


def stepAI(action, stateInfo):
    obs, reward, done, info = env.step(action)
    if done:
        resetAI(stateInfo)
    else:
        redraw(obs, False)


def key_handler(event):
    print('pressed', event.key)
    if event.key == 'escape':
        window.close()
        return
    if event.key == 'backspace':
        resetManual()
        return
    if event.key == 'left':
        stateInfo = stepManual(env.actions.left)
        if stateInfo is not None:
            resetAI(stateInfo)
        return
    if event.key == 'right':
        stateInfo = stepManual(env.actions.right)
        if stateInfo is not None:
            resetAI(stateInfo)
        return
    if event.key == 'up':
        stateInfo = stepManual(env.actions.forward)
        if stateInfo is not None:
            print("test!!!!!!!!!!!!!!!!!!!")
            resetAI(stateInfo)
        return
    if event.key == ' ':
        stateInfo = stepManual(env.actions.toggle)
        if stateInfo is not None:
            resetAI(stateInfo)
        return
    if event.key == 'pageup':
        stateInfo = stepManual(env.actions.pickup)
        if stateInfo is not None:
            resetAI(stateInfo)
        return
    if event.key == 'pagedown':
        stateInfo = stepManual(env.actions.drop)
        if stateInfo is not None:
            resetAI(stateInfo)
        return
    if event.key == 'enter':
        stateInfo = stepManual(env.actions.done)
        if stateInfo is not None:
            resetAI(stateInfo)
        return


def saveState(env):
    door_x = env.door_pos[0]
    door_y = env.door_pos[1]
    door = env.grid.get(door_x, door_y)
    state = {
        "grid": env.grid,
        "grid_encoding": env.grid.encode().tolist(),
        "envGrid": env.grid.copy(),
        "splitIdx": env.splitIdx,
        "doorIdx": env.doorIdx,
        "door_pos": env.door_pos,
        "key_pos": env.key_pos,
        "agent_pos": env.agent_pos,
        "agent_dir": env.agent_dir,
        "carrying": env.carrying,
        "step_count": env.step_count,
        "is_locked": door.is_locked,
        "is_open": door.is_open
    }
    return state


def loadState(state, env):
    env.envs[0].agent_pos = state["agent_pos"]
    env.envs[0].agent_dir = state["agent_dir"]
    env.envs[0].grid = state["envGrid"]
    door_x = env.envs[0].door_pos[0]
    door_y = env.envs[0].door_pos[1]
    door = env.envs[0].grid.get(door_x, door_y)
    door.is_locked = state["is_locked"]
    door.is_open = state["is_open"]
    env.envs[0].grid.set(door_x, door_y, door)
    return env


parser = argparse.ArgumentParser()
parser.add_argument(
    "--env", required=True,
    help="gym environment to load (REQUIRED)",
    default='MiniGrid-MultiRoom-N6-v0'
)
parser.add_argument(
    "--model", required=True,
    help="name of the trained model",
    default=""
)
parser.add_argument(
    "--episodes", required=True,
    help="number of episodes to predict for the agent",
    default=1
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)
parser.add_argument(
    '--agent_view',
    default=False,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)


# MANUAL CONTROL *********************
args = parser.parse_args()
env = gym.make(args.env)
if args.agent_view:
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
envOriginal = env
# ************************************


# Load the window
window = Window('gym_minigrid - ' + args.env)
window.reg_key_handler(key_handler)

# Initialize
uniqueStartStates = {}
stateInfo = resetManual()

# Blocking event loop
window.show(block=True)


# AI AGENT *******************************
args.episodes = 1
envs = []
envs.append(envOriginal)
env = ParallelEnv(envs)
# ****************************************


# Load the window
window = Window('gym_minigrid - ' + args.env)

# Load agent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                    device=device, argmax=True, num_envs=1,
                    use_memory=False, use_text=False)

# Initialize
resetAI(stateInfo)
obss = [env.envs[0].gen_obs()]
print("obs:"+str(obss))


# Construct AI Replay
prevAction = -1
done = False
frames = []
while not done:
    actions = agent.get_actions(obss)
    frames.append(numpy.moveaxis(env.envs[0].render("rgb_array"), 2, 0))
    obss, rewards, dones, _ = env.step(actions)
    done = dones[0]
    print("actions:" + str(actions))
    print("done:"+str(done))
print("Saving gif... ", end="")
write_gif(numpy.array(frames), "Results/AI_Playback/" + str(args.env) + "_" + str(args.model) + "_" + str(args.seed) + "_" + str(1) + "_actions.gif", fps=1/1)

# Blocking event loop
window.show(block=True)




"""
def reset(isManual, stateInfo=None):
    if args.seed != -1:
        env.seed(args.seed)

    if int(args.episodes) > 0:
        valid = False
        while not valid:
            if isManual:  
                obs = env.reset(None)
                if env.hash() not in uniqueStartStates.keys():
                    valid = True
                    uniqueStartStates[env.hash()] = True
                    print("unique: " + str(uniqueStartStates))
                    args.episodes = str(int(args.episodes) - 1)
                    state = saveState(env)

            else:
                obs = env.envs[0].reset(None)
                
                # set the state of the environment
                
                env.envs[0].agent_pos = stateInfo["agent_pos"]
                env.envs[0].agent_dir = stateInfo["agent_dir"]
                print("setting direction to " + str(stateInfo["agent_dir"]))

                env.envs[0].grid = stateInfo["envGrid"]
                door_x = env.envs[0].door_pos[0]
                door_y = env.envs[0].door_pos[1]
                door = env.envs[0].grid.get(door_x, door_y)
                door.is_locked = stateInfo["is_locked"]
                door.is_open = stateInfo["is_open"]
                env.envs[0].grid.set(door_x, door_y, door)
            
                print("env.hash(): " + str(env.envs[0].hash()))

                if env.envs[0].hash() not in uniqueStartStates.keys():
                    valid = True
                    uniqueStartStates[env.envs[0].hash()] = True
                    print("unique: " + str(uniqueStartStates))
                    args.episodes = str(int(args.episodes) - 1)
                
        if hasattr(env, 'mission'):
            print('Mission: %s' % env.mission)
            window.set_caption(env.mission)

        redraw(obs, isManual)
        return obs
    else:
        window.close()
        return None
"""