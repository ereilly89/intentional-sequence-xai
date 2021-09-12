#!/usr/bin/env python3
import argparse
from array2gif import write_gif
import gym
import gym_minigrid
from gym_minigrid.window import Window
from gym_minigrid.wrappers import *
import numpy
import os.path
import time
import torch
from torch_ac.utils.penv import ParallelEnv
from typing import Counter
import utils
import pandas as pd


def redraw(env, img, isManual):
    if not args.agent_view:
        if isManual:
            img = env.render('rgb_array', tile_size=args.tile_size)
        else:
            img = env.envs[0].render('rgb_array', tile_size=args.tile_size)

    window.show_img(img)


def resetManual(env): # returns a stateInfo dict
    if args.seed != -1:
        env.seed(args.seed)

    valid = False
    while not valid:
        obs = env.reset(None)
        if env.hash() not in uniqueStartStates.keys():
            valid = True
            uniqueStartStates[env.hash()] = True
            print("unique: " + str(uniqueStartStates))
            state = saveState(env)
    redraw(env, obs, True)
    return state


def resetAI(stateInfo, env): # returns a ParallelEnv
    print("\n\nRESET AI********************\n\n")

    envs = []
    envs.append(env)
    env = ParallelEnv(envs)
    env = loadState(stateInfo, env)

    if args.seed != -1:
        env.seed(args.seed)

    print("\n\nepisodes: " + str(args.episodes))
    obs = env.envs[0].reset(None)
        
    # set the state of the environment
    env.envs[0].agent_pos = stateInfo["agent_pos"]
    env.envs[0].agent_dir = stateInfo["agent_dir"]
    redraw(env, obs, False)

    return env
        

def stepManual(action): 
    obs, reward, done, info = env.step(action)
    print("DONE: " + str(done))
    if done:
        return stateInfo
    else:
        redraw(env, obs, True)
        return None


def key_handler(event):
    print('pressed', event.key)
    if event.key == 'escape':
        window.close()
        return
    """
    if event.key == 'backspace':
        resetManual()
        return
    """
    if event.key == 'left':
        stateInfo = stepManual(env.actions.left)
        count["count"] = count["count"] + 1
        print("\nPRESSED LEFT...\n")
        if stateInfo is not None:
            window.close()
            env.seed(args.seed)
            environment = resetAI(stateInfo, env)
            writeAgentPlayback(environment)
            
        return

    if event.key == 'right':
        stateInfo = stepManual(env.actions.right)
        count["count"] = count["count"] + 1
        print("\nPRESSED RIGHT...\n")
        if stateInfo is not None:
            window.close()
            env.seed(args.seed)
            environment = resetAI(stateInfo, env)
            writeAgentPlayback(environment)
        return

    if event.key == 'up':
        stateInfo = stepManual(env.actions.forward)
        count["count"] = count["count"] + 1
        print("\nPRESSED UP...\n") 
        print("stateInfo: " + str(stateInfo))
        if stateInfo is not None:
            window.close()
            env.seed(args.seed)
            environment = resetAI(stateInfo, env)
            writeAgentPlayback(environment)
        return

    if event.key == ' ':
        stateInfo = stepManual(env.actions.toggle)
        count["count"] = count["count"] + 1
        print("\nPRESSED SPACE...\n")
        if stateInfo is not None:
            window.close()
            env.seed(args.seed)
            environment = resetAI(stateInfo, env)
            writeAgentPlayback(environment)
        return

    if event.key == 'pageup':
        stateInfo = stepManual(env.actions.pickup)
        count["count"] = count["count"] + 1
        print("\nPRESSED PAGEUP...\n")
        if stateInfo is not None:
            window.close()
            env.seed(args.seed)
            environment = resetAI(stateInfo, env)
            writeAgentPlayback(environment)
        return
        
    if event.key == 'pagedown':
        stateInfo = stepManual(env.actions.drop)
        count["count"] = count["count"] + 1
        print("\nPRESSED PAGEDOWN...\n")
        if stateInfo is not None:
            window.close()
            env.seed(args.seed)
            environment = resetAI(stateInfo, env)
            writeAgentPlayback(environment)
        return

    """
    if event.key == 'enter':
        stateInfo = stepManual(env.actions.done)
        if stateInfo is not None:
            resetAI(stateInfo)
        return
    """

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


def writeAgentPlayback(env):
    obss = [env.envs[0].gen_obs()]

    # Construct AI Replay
    prevAction = -1
    done = False
    frames = []
    AI_STEPS = 0
    while not done:
        AI_STEPS = AI_STEPS + 1
        actions = agent.get_actions(obss)
        frames.append(numpy.moveaxis(env.envs[0].render("rgb_array"), 2, 0))
        obss, rewards, dones, _ = env.step(actions)
        done = dones[0]

    print("AI Steps: " + str(AI_STEPS))
    #print("Manual Steps: " + str(manualSteps))
    #difference = AI_STEPS - manualSteps
    #print("Difference: " + str(difference))
    
    if args.model == "Model_A" or args.model == "Model_B" or args.model == "Model_C":
        message = args.model + ", " + str(count["count"]) + ", " + str(AI_STEPS) #+ ", Manual Steps: " + str(manualSteps) + ", Difference: " + str(difference)
        print(message, file=open("Results/intentional_evaluation.csv", "a"))
    elif args.model == "Model_D" or args.model == "Model_E" or args.model == "Model_F":
        message = args.model + ", " + str(count["count"]) + ", " + str(AI_STEPS) #+ ", Manual Steps: " + str(manualSteps) + ", Difference: " + str(difference)
        print(message, file=open("Results/random_evaluation.csv", "a"))
    else:
        print("Invalid model, try 'Model_A', 'Model_B', ..., or 'Model_F'")
    print("Saving gif... ", end="")
    write_gif(numpy.array(frames), "Results/AI_Playback/" + str(args.env) + "_" + str(args.model) + "_" + str(args.seed) + "_" + str(args.episodes) + "_actions.gif", fps=1/1)


def is_integer(n):
    try:
        int(n)
        return True
    except ValueError:
        return False


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

# Load agent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                    device=device, argmax=False, num_envs=1,
                    use_memory=False, use_text=False)

if not os.path.isfile("Results/intentional_evaluation.csv"):
    print("model, prediction, actual", file=open("Results/intentional_evaluation.csv", "w+"))

if not os.path.isfile("Results/intentional_correlation.csv"):
    print("model, intentionality_user, intentionality_in_order", file=open("Results/intentional_correlation.csv", "w+"))

if not os.path.isfile("Results/random_evaluation.csv"):
    print("model, prediction, actual", file=open("Results/random_evaluation.csv", "w+"))

if not os.path.isfile("Results/random_correlation.csv"):
    print("model, intentionality_user, intentionality_in_order", file=open("Results/random_correlation.csv", "w+"))


#if not os.path.isfile("Results/random_intentionality.csv"):
#    print("")

for i in range(int(args.episodes)):

    args.seed = args.seed + i*100 + 99

    # Load the window (manual control)
    window = Window('gym_minigrid - ' + args.env)
    window.reg_key_handler(key_handler)
  
    # Initialize
    uniqueStartStates = {}
    count = {}
    count["count"] = 0

    stateInfo = resetManual(env)

    # Blocking event loop
    window.show(block=True)

    args.episodes = str(int(args.episodes) - 1)

budget = 10
unranked = {}
for i in range(budget):
    unranked[i] = True

intentionality_user = []
for i in range(10):
    valid = False
    while not valid:

        print("\n")
        for seq in unranked.keys():
            print("[" + str(seq) + "]: sequence #" + str(seq))
        print("\n")

        if i == 0:
            theInput = input("Which sequence was the most informative to you in understanding the agents decision-making process?\n")
        else:
            theInput = input("Which sequence was the next most informative to you in understanding the agents decision-making process?\n")
      
        if is_integer(theInput):
            intInput = int(theInput)

            if intInput in unranked.keys():
                unranked.pop(intInput, None)

                if args.model == "Model_A" or args.model == "Model_B" or args.model == "Model_C": 
                    intentions = pd.read_csv("Results/intentional/" + args.model + "/intentionality.csv")
                elif args.model == "Model_D" or args.model == "Model_E" or args.model == "Model_F": 
                    intentions = pd.read_csv("Results/random/" + args.model + "/random_intentionality.csv")
                else:
                    print("Invalid model, try 'Model_A', 'Model_B', ..., or 'Model_F'")

                intentionsDict = intentions.to_dict()
                intentVal = intentionsDict["Intentionality"][intInput]
                intentionality_user.append(intentVal)
                valid = True

        if not valid:
            print("\n------------------------------------------\nPlease enter a valid sequence number.")

userIntentRank = pd.Series(intentionality_user)
actualIntentRank = pd.Series(sorted(intentionality_user, reverse=True))

if args.model == "Model_A" or args.model == "Model_B" or args.model == "Model_C":
    for i in range(len(userIntentRank)):
        print(args.model + "," + str(userIntentRank[i]) + "," + str(actualIntentRank[i]), file=open("Results/intentional_correlation.csv", "a"))
elif args.model == "Model_D" or  args.model == "Model_E" or args.model == "Model_F":
    for i in range(len(userIntentRank)):
        print(args.model + "," + str(userIntentRank[i]) + "," + str(actualIntentRank[i]), file=open("Results/random_correlation.csv", "a"))
else:
    print("Invalid model, try 'Model_A', 'Model_B', ..., or 'Model_F'")


#print("Pearson Correlation: " + str(userIntentRank.corr(actualIntentRank)))
#print("Spearman Correlation: " + str(userIntentRank.corr(actualIntentRank, method='spearman')))