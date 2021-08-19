
# was visualize.py

import argparse
import time
import numpy
import torch
import utils
import os
from os.path import exists
from array2gif import write_gif
from gym_minigrid.minigrid import Door


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
        5: "opens the door",
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

    def explain(sequence, s):
        frames = []
        description = []
        actions = getActionSequence(sequence)

        explainFile = gif + str(s) + "_explanation.txt"
        if exists(explainFile):
            os.remove(explainFile) 

        startState = getStartState(sequence)
        env = envMap[startState]
        envHash = ""
        
        env = envMap[getStartState(sequence)]
        stateInfo = statesInfo[getStartState(sequence)]

        env.agent_pos = stateInfo["agent_pos"]
        env.agent_dir = stateInfo["agent_dir"]
        env.grid = stateInfo["envGrid"]

        door_x = env.door_pos[0]
        door_y = env.door_pos[1]
        door = env.grid.get(door_x, door_y)
        door.is_locked = stateInfo["is_locked"]
        door.is_open = stateInfo["is_open"]
        env.grid.set(door_x, door_y, door)


        envHash = env.hash()
        #door = new Door() stateInfo["door_is_open"]
        #env.grid.set(env, env.door_pos[0], env.door_pos[1], door)

        obs, reward, done, _ = env.step(0)
        obs, reward, done, _ = env.step(1)

        env = envMap[getStartState(sequence)]
        stateInfo = statesInfo[getStartState(sequence)]

        env.agent_pos = stateInfo["agent_pos"]
        env.agent_dir = stateInfo["agent_dir"]

       
        # env.grid = stateInfo["envGrid"]

        sentence = "The agent starts pointed " + str(TO_DIR[env.agent_dir])

        print("agent dir, env: " + str(env.agent_dir))
        print("grid.encode().tolist(), env: " + str(env.grid.encode().tolist()))
        print("agent pos: " + str(env.agent_pos))
        print("agent dir: " + str(env.agent_dir))
        print("key pos: " + str(env.key_pos))
        print("door pos: " + str(env.door_pos))
        print("carrying: " + str(env.carrying)) #might have to change

        print("stateInfo: " + str(stateInfo))

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
        print("2- " + str(env.hash()))

        for action in actions:
            frames.append(numpy.moveaxis(env.render("rgb_array"), 2, 0))
            obs, reward, done, _ = env.step(action)

            sentence = "Then, the agent " + TO_ACT[action]
            gridList = obs["image"] #env.grid.encode().tolist()
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

            print("2- " + str(env.hash()))
            #agent.analyze_feedback(reward, done)
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
            print(sentence)

    for s in sequences:

        # Set seed for all randomness sources

        # utils.seed(seed)


        # Set device

        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(f"Device: {device}\n")
        

        # Load environment
        #env = utils.make_env(environment, seed) # args.env args.seed
        
        """
        env = envMap[getStartState(sequences[s]["sequence"])]
        stateInfo = statesInfo[getStartState(sequences[s]["sequence"])]
        env.set_state(stateInfo)
        """

        print(str(s) + " : " + str(sequences[s]["sequence"]))
        # print("keyObs:"+str(images[getStartState(sequences[s]["sequence"])]))
        

        """
        print("Environment loaded\n")


        # Load agent
        
        model_dir = utils.get_model_dir(model) #args.model
        agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                            device=device, argmax=argmax, use_memory=memory, use_text=text)
       
        print("Agent loaded\n")
         """

        # Create a window to view the environment

        #env.render('human')


        # Create a GIF for each explanation sequence

        frames = []

        #actions = getActionSequence(sequences[s]["sequence"])

        #print("debug: "+str(sequences[s]["sequence"]))


        """
        states = getStateSequence(sequences[s]["sequence"])
    
        for state in states:
            env = envMap[str(state)]
            env.render('human')

            
            # print("agent_pos: " + str(env.agent_pos))
            # print("door_pos" + str(env.door_pos))
            # print("key_pos" + str(env.key_pos))
            

            stateInfo = statesInfo[str(state)]
            print("1- " + str(env.hash()))
            #env.reset(stateInfo=stateInfo)

            # env = env.set_state(envObj) #{'agent_pos':(x, y), ...etc} #verify state is exactly what we need it to be

            env.render('human')
            if gif:
                frames.append(numpy.moveaxis(env.render("rgb_array"), 2, 0))

            # obs, reward, done, _ = env.step(action)
            #print("OBS:"+str(obs))
            #print("action:"+str(action))
            # agent.analyze_feedback(reward, done)

            if env.window.closed:
                break

        if env.window.closed:
            break

        try:
            if gif:
                print("Saving gif... ", end="")
                write_gif(numpy.array(frames), gif + str(s) + "_states.gif", fps=1/pause)
                print("Done.")
        except ValueError:
            print("sequence-> " + str(sequences[s]["sequence"]))
            print("value error.")

        """


        explain(sequences[s]["sequence"], s)
        """
        frames = []
        actions = getActionSequence(sequences[s]["sequence"])

        explanation = []

        env = envMap[getStartState(sequences[s]["sequence"])]
        stateInfo = statesInfo[getStartState(sequences[s]["sequence"])]
        #env.reset(stateInfo)
        env.render('human')
        print("2- " + str(env.hash()))
        for action in actions:
            
            #env.render('human')
            if gif:
                frames.append(numpy.moveaxis(env.render("rgb_array"), 2, 0))

            obs, reward, done, _ = env.step(action)
            print("2- " + str(env.hash()))
            #print("OBS:"+str(obs))
            #print("action:"+str(action))
            agent.analyze_feedback(reward, done)

            if done or env.window.closed:
                break

        frames.append(numpy.moveaxis(env.render("rgb_array"), 2, 0))
        if env.window.closed:
            break

        try:
            if gif:
                print("Saving gif... ", end="")
                write_gif(numpy.array(frames), gif + str(s) + "_actions.gif", fps=1/pause)
                print("Done.")
        except ValueError:
            print("sequence-> " + str(sequences[s]["sequence"]))
            print("value error.")
        """
