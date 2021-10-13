
# was visualize.py

import argparse
import time
import numpy
import torch
import utils
import os
from os.path import exists
from array2gif import write_gif
from gym_minigrid.minigrid import Door, Key


def visualize(sequences, images, envMap, environment, model, argmax, seed, memory, text, episodes, pause, gif, model_dir, agent, statesInfo):

    TO_DIR = {
        0: "right",
        1: "down",
        2: "left",
        3: "up"
    }

    TO_OBJ = {
        4: "door",
        5: "key",
        8: "goal"
    }

    TO_COLOR = {
        0: "red",
        1: "green",
        2: "blue",
        3: "purple",
        4: "yellow",
        5: "grey"
    }

    TO_ACT = {
        0: "turns left",
        1: "turns right",
        2: "moves forward",
        3: "picks up the key",
        4: "drops the key",
        5: "toggles the door",
        6: "is done"
    }

    def getStartState(sequence):
        x = sequence[0]
        startState = x[1:]
        #print("startState:"+startState)
        return startState
    
    def getStateSequence(sequence):
        states = []
        for i in range(len(sequence)):
            if i % 2 == 0:
                x = sequence[i]
                states.append(str(x[1:]))
        return states
        
    def getActionSequence(sequence):
        actions = []
        for i in range(len(sequence)):
            if i % 2 != 0:
                x = sequence[i]
                actions.append(int(x[1:]))
        return actions


    def loadState(state, env, startState, statesInfo):

        # set the agent
        env.agent_pos = state["agent_pos"]
        env.agent_dir = state["agent_dir"]

        # set the door
        door_x = env.door_pos[0]
        door_y = env.door_pos[1]
        door = env.grid.get(door_x, door_y)
        door.is_locked = state["is_locked"]
        door.is_open = state["is_open"]
        env.grid.set(door_x, door_y, door)

        # set the key
        key_x = env.key_pos[0]
        key_y = env.key_pos[1]
        env.carrying = state["carrying"]
        if env.carrying is not None:
            for j in range(env.grid.height):
                for i in range(env.grid.width):
                    c = env.grid.get(i, j)
                    if c is not None:
                        if c.type == 'key':
                            env.grid.set(i, j, None)
        else:
            key = Key("yellow")
            #print("grid" + str(env.grid))
            #print("KEY_X: " + str(key_x))
            #print("KEY_Y: " + str(key_y))

            envHash = None
            #while envHash != startState:

            for j in range(env.grid.height):
                for i in range(env.grid.width):
                    c = env.grid.get(i, j)
                    #print("c:"+str(c))
                    if c is not None:
                        if c.type == 'key':
                            env.grid.set(i, j, None)
            
            gridList = statesInfo[startState]["grid_encoding"]
            width = 0

            for a in gridList:
                height = 0

                for b in a:
                    theTuple = gridList[width][height]

                    if theTuple[0] == 5:
                        env.grid.set(width, height, key)
                        break

                    height += 1

                width += 1

            #print("\nenv.hash(): " + str(env.hash()) + ", startState: " + str(startState))
            #raise Exception("does this even get hit?")
            if env.hash() != startState:
                print("\n\nTarget: " + str(statesInfo[startState]))
                #print("Actual: " + str(statesInfo[env.hash()]))
                print("\n")
                print("startState: " + str(startState))
                print("envHash: " + str(envHash))
                print("grid_encoding -> " + str(env.grid.encode().tolist()))
                print("agent_pos -> " + str(env.agent_pos))
                print("agent_dir -> " + str(env.agent_dir))
                print("key_pos -> " + str(env.key_pos))
                print("door_pos -> " + str(env.door_pos))
                print("carryingKey -> " + str(env.carrying)+"\n\n")

            #env.grid.set(key_x, key_y, key)
            #env.put_obj(key, key_x, key_y)
        return env


    def explain(sequence, s):
        frames = []
        description = []
        actions = getActionSequence(sequence)
        print("s: " + str(s))
        print("sequence: " + str(sequence))

        explainFile = gif + str(s) + "_explanation.txt"
        if exists(explainFile):
            os.remove(explainFile) 

        startState = getStartState(sequence)
        env = envMap[startState]

        envHash = ""
        stateInfo = statesInfo[startState]
        env = loadState(stateInfo, env, startState, statesInfo)
       
        if env.carrying is not None:
            print(str(True))
        else:
            print(str(False))

        envHash = env.hash()

        if startState != envHash:
            print("stateInfo: " + str(stateInfo))
            print("startState: " + str(startState))
            print("envHash: " + str(envHash))
            print("grid_encoding -> " + str(env.grid.encode().tolist()))
            print("agent_pos -> " + str(env.agent_pos))
            print("agent_dir -> " + str(env.agent_dir))
            print("key_pos -> " + str(env.key_pos))
            print("door_pos -> " + str(env.door_pos))
            print("carryingKey -> " + str(env.carrying))
            raise Exception("mismatch exception 1")
       
        obs, reward, done, _ = env.step(0)
        obs, reward, done, _ = env.step(1)

        env = envMap[startState]
        stateInfo = statesInfo[startState]
        env = loadState(stateInfo, env, startState, statesInfo)

        if startState != envHash:
            print("stateInfo: " + str(stateInfo))
            print("envHash: " + str(envHash))
            raise Exception("mismatch exception 2")

        sentence = "The agent starts pointed " + str(TO_DIR[env.agent_dir])

        gridList = obs["image"] # env.grid.encode().tolist()
        count = 0
        visited = {}
        keyMessage = False
        for i in range(len(gridList)):
            for j in range(len(gridList[i])):
                observ = gridList[i][j]

                if observ[0] != 0 and observ[0] != 1 and observ[0] != 2 and observ[0] != 3:
                    
                    if observ[0] != 5 or (observ[0] == 5 and env.carrying is None): # dont include key here
                        if count == 0:
                            sentence = sentence + ", with the " + TO_COLOR[observ[1]] + " " + TO_OBJ[observ[0]]
                        else:
                            sentence = sentence + " and " + TO_COLOR[observ[1]] + " " + TO_OBJ[observ[0]]
                        count = count + 1
                    else:
                        keyMessage = True
                    visited[TO_OBJ[observ[0]]] = True
        
        if count > 0:
            sentence = sentence + " in sight"

        if env.carrying is not None:
            sentence = sentence + " and is carrying the key"

        sentence = sentence + ". "
        description.append(sentence)

        env.render('human')

        for action in actions:
            frames.append(numpy.moveaxis(env.render("rgb_array"), 2, 0))

            isCarrying = False
            if env.carrying is not None:
                isCarrying = True

            print("str(action): " + str(action))
            obs, reward, done, _ = env.step(action)

            #if s == 1:
            #    print("isCarrying, " + str(isCarrying))
            #    print("action, " + str(action))

            if isCarrying == True and action == 3:
                sentence = "Then, the agent tries to pick up the key but already has it"
            else:
                sentence = "Then, the agent " + TO_ACT[action]

            gridList = obs["image"]
            count = 0

            for i in range(len(gridList)):
                for j in range(len(gridList[i])):
                    observ = gridList[i][j]

                    # if a special observation
                    if observ[0] != 0 and observ[0] != 1 and observ[0] != 2 and observ[0] != 3 and TO_OBJ[observ[0]] not in visited.keys():
                        if count == 0:
                            sentence = sentence + " to see the " + TO_OBJ[observ[0]]
                        else:
                            sentence = sentence + " and " + TO_OBJ[observ[0]]
                        count = count + 1

                        visited[TO_OBJ[observ[0]]] = True
            
            sentence = sentence + ". "
            description.append(sentence)

            if done or env.window.closed:
                break

        frames.append(numpy.moveaxis(env.render("rgb_array"), 2, 0))
        if env.window.closed:
            return

        try:
            if gif:
                print("Saving gif... ", end="")
                write_gif(numpy.array(frames), gif + str(s) + "_actions.gif", fps=1/pause)
                print("Done.")
        except ValueError:
            print("sequence-> " + str(sequence))
            print("value error.")

        for sentence in description:
            print(sentence, file=open(gif + str(s) + "_explanation.txt", "a"))
            #print(sentence)
    
        #if s == 1:
        #    print("meh")
    
    for s in sequences:
        frames = []
        explain(sequences[s]["sequence"], s)
