import argparse
import copy
from scripts.intentions_IRL import getIntentionalStates
import time
import torch
from torch_ac.utils.penv import ParallelEnv
import hashlib
import utils
import os
from scripts.intentions_RL import getIntentionalStates, buildSequences, buildRandomSequences
from scripts.visualize_sequences import visualize
from gym_minigrid.minigrid import Door, Key

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
        env.grid.set(key_x, key_y, None)
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


def main():
    print("main...")
    # Parse arguments

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True,
                        help="name of the environment (REQUIRED)")
    parser.add_argument("--model", required=True,
                        help="name of the trained model (REQUIRED)")
    parser.add_argument("--episodes", type=int, default=100,
                        help="number of episodes of evaluation (default: 100)")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed (default: 0)")
    parser.add_argument("--procs", type=int, default=16,
                        help="number of processes (default: 16)")
    parser.add_argument("--argmax", action="store_true", default=False,
                        help="action with highest probability is selected")
    parser.add_argument("--worst-episodes-to-show", type=int, default=10,
                        help="how many worst episodes to show")
    parser.add_argument("--memory", action="store_true", default=False,
                        help="add a LSTM to the model")
    parser.add_argument("--text", action="store_true", default=False,
                        help="add a GRU to the model")
    args = parser.parse_args()

    # Set seed for all randomness sources

    utils.seed(args.seed)

    # Set device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load environments
    print("loading environments...")
    envs = []
    for i in range(args.procs):
        env = utils.make_env(args.env, args.seed + 10000 * i)
        envs.append(env)
        print("appending env " + str(env))
    env = ParallelEnv(envs)
    
    """
    env = utils.make_env(args.env, args.seed)
    print("env:"+str(env))
    print("Environments loaded\n")
    """
    # Load agent
    print("loading model and agent...")
    model_dir = utils.get_model_dir(args.model)
    agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                        device=device, argmax=args.argmax, num_envs=args.procs,
                        use_memory=args.memory, use_text=args.text)
    print("Agent loaded\n")

    # Initialize logs

    logs = {"num_frames_per_episode": [], "return_per_episode": []}

    # Run agent

    start_time = time.time()

    obss = env.reset()

    log_done_counter = 0
    log_episode_return = torch.zeros(args.procs, device=device)
    log_episode_num_frames = torch.zeros(args.procs, device=device)


    graph = {}
    frequencies = {}
    images = {}
    envMap = {}
    statesInfo = {}
    count = 0
    numEdges = 0
    while log_done_counter < args.episodes:
        
        isDone = False
        keys = obss
        #print("obss:"+str(obss))
        actions = agent.get_actions(obss)

        startEnvStateID = env.envs[0].hash()
        envBeforeAction = env.envs[0]
        print("startEnvStateID = " + startEnvStateID)


        if startEnvStateID not in images.keys():
            images[startEnvStateID] = keys[0] 

        if startEnvStateID not in envMap.keys():
            envMap[startEnvStateID] = copy.deepcopy(env.envs[0])  # print(print("copy:"+str(copy.deepcopy(env.unwrapped.clone_full_state())))) #copy.deepcopy(env)
            #print("deep copy!!!"+ str(envMap[stateIndex]))

        if startEnvStateID not in statesInfo.keys():

            #get the is_locked and is_open attributes of Door
            #and set them in the info dict below
            door_x = env.envs[0].door_pos[0]
            door_y = env.envs[0].door_pos[1]
            door = env.envs[0].grid.get(door_x, door_y)

            print("door_x: " + str(door_x))
            print("door_y" + str(door_y))
            print("door:" + str(door))
            print("door.is_locked" + str(door.is_locked))
            print("door.is_open" + str(door.is_open))
            print("seed!:"+str(env.envs[i].theSeed))
            info = {
                "grid": env.envs[0].grid,
                "seed": env.envs[0].theSeed,
                "grid_encoding": env.envs[0].grid.encode().tolist(),
                "envGrid": env.envs[0].grid.copy(),
                "splitIdx": env.envs[0].splitIdx,
                "goal_pos": env.envs[0].goal_pos,
                "doorIdx": env.envs[0].doorIdx,
                "door_pos": env.envs[0].door_pos,
                "key_pos": env.envs[0].key_pos,
                "agent_pos": env.envs[0].agent_pos,
                "agent_dir": env.envs[0].agent_dir,
                "carrying": env.envs[0].carrying,
                "step_count": env.envs[0].step_count,
                "is_locked": door.is_locked,
                "is_open": door.is_open
            }
            statesInfo[startEnvStateID] = info

            #if startEnvStateID == "960e88ba821d7fd6": # "6c673edb7df77cd5":
            #    raise Exception('info:'+str(info))

            #print("info['agent_pos'][0]: " + str(info["agent_pos"][0]))
            #print("info['agent_pos'][1]: " + str(info["agent_pos"][1]) + "\n")
            #print("info['goal_pos'][0]: " + str(info["goal_pos"][0]))
            #print("info['goal_pos'][1]: " + str(info["goal_pos"][1]) + "\n")
            #raise Exception("meh.")
            #if info["agent_pos"][0] == info["goal_pos"][0] and info["agent_pos"][1] == info["goal_pos"][1]:
            #    raise Exception("REACHED GOAL.")

        print("\n statesInfo[startEnvStateID] (before): " + str(statesInfo[startEnvStateID]))

        
        the_agent_pos = env.envs[0].agent_pos
        the_goal_pos = env.envs[0].goal_pos

        print("actions[0]: " + str(actions[0]))
        if (the_agent_pos[0] == 3 and the_agent_pos[1] == 2 and info["agent_dir"] == 1 and actions[0] == 2) or (the_agent_pos[0] == 2 and the_agent_pos[1] == 3 and info["agent_dir"] == 0 and actions[0] == 2):
            print("dones:"+str(dones))
            print("rewards:"+str(rewards))
            dones = (True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False)
            rewards = (1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            endEnvStateID = "gooooooooooooal"
            obss = env.reset()
            #raise Exception("they are equal (1).")

        
        #check for intermediate reward (picking up the key)
        elif info["carrying"] is None and actions[0] == 3:
            obss, rewards, dones, _ = env.step(actions)
            if env.envs[0].carrying is not None:
                rewards = (0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            endEnvStateID = env.envs[0].hash()

        #check for intermediate reward (dropping the key)
        elif info["carrying"] is not None and actions[0] == 4: 
            obss, rewards, dones, _ = env.step(actions)
            if env.envs[0].carrying is None:
                rewards = (-0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            endEnvStateID = env.envs[0].hash()

        else:
            obss, rewards, dones, _ = env.step(actions)
            endEnvStateID = env.envs[0].hash()

        #print("statesInfo (after): " + str(statesInfo[env.envs[0].hash()]))
        



        """
        print("\nagent_pos: " + str(agent_pos))
        print("goal_pos: " + str(goal_pos) + "\n")
        print("agent_pos[0]: " + str(agent_pos[0]))
        print("agent_pos[1]: " + str(agent_pos[1]))
        print("goal_pos[0]: " + str(goal_pos[0]))
        print("goal_pos[1]: " + str(goal_pos[1]))
        """


        """
        if dones[0] == True:
            beforeGoal = statesInfo[startEnvStateID]
            beforeGoal["agent_pos"] = beforeGoal["goal_pos"]
            onGoal = beforeGoal
            newEnv = env.envs[0]
            print("\n\nIS THIS WHERE IT FAILS?\n")
            newEnv = loadState(onGoal, env.envs[0], startEnvStateID, statesInfo)

            endEnvStateID = newEnv.hash()
        else:
        """

        #endEnvStateID = env.envs[0].hash() 
        



        print("endEnvStateID: " + endEnvStateID)
        #if dones[0] == True:
        #    continue #isDone = True #continue

        for i in range(args.procs):
            testKey = keys[i]
            #print("keys[i]"+str(keys[i]))

            print("\n"+str(i)+"\n")

            stateIndex = startEnvStateID #getHash(keys[i])
            endIndex = endEnvStateID #getHash(obss[i])


            if dones[0] == True:
                print("actions[i]: " + str(actions[i]))
                #raise Exception("")

            actionIndex = actions[i]

            value = {"obs":    endIndex,
                     "reward": rewards[i]}
            print("REWARD: " + str(rewards[i]))

            #if rewards[i] != 0:
                #print()
                #raise Exception("Ayyyyyyyyy rewards[i]: " + str(rewards[i]))

            #if stateIndex == "d83b1f853be4e1be": #or stateIndex == "03aaa7557192803e":
            #    print("STATESINFO: " + str(statesInfo[stateIndex]))
            #    raise Exception("just a value")
            
            startState = statesInfo[stateIndex]
            print(str(startState))

            """
            endEnv = env.envs[0]
            isValid = True
            if endEnv.agent_pos[0] == endEnv.goal_pos[0] and endEnv.agent_pos[1] == endEnv.goal_pos[1]:
                isValid = False
                raise Exception("stop")
            """

            # graph
            if stateIndex in graph.keys() :
               
                edges = graph[stateIndex] 
               
                if actionIndex in edges.keys():
                    freq = edges[actionIndex]["prob"] 
                    edges[actionIndex]["prob"] = freq + 1
                else:
                    value["prob"] = 1
                    edges[actionIndex] = value
                    numEdges = numEdges + 1
                  
                graph[stateIndex][actionIndex] = edges[actionIndex]
                

            else:
                value["prob"] = 1
                graph[stateIndex] = {}
                graph[stateIndex][actionIndex] = value

            print("graph[stateIndex][actionIndex]:"+str(graph[stateIndex][actionIndex]))

            # get frequencies
            if stateIndex in frequencies.keys():
                freq = frequencies[stateIndex]
                frequencies[stateIndex] = freq + 1
            else:
                frequencies[stateIndex] = 1

            break #temporary? no

        count = count + 1   


        if dones[0] == True:
            continue
        
        agent.analyze_feedbacks(rewards, dones)

        log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
        log_episode_num_frames += torch.ones(args.procs, device=device)

        for i, done in enumerate(dones):
            if done:
                log_done_counter += 1
                logs["return_per_episode"].append(log_episode_return[i].item())
                logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())

        mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
        log_episode_return *= mask
        log_episode_num_frames *= mask

    #raise Exception("just an exception, get rid of later")
    

    # convert frequencies to probabilities
    reverseGraph = {}
    for u in graph.keys():
        for actionIndex in graph[u].keys():
            #print("edges:"+str(edges))
            value = graph[u][actionIndex]

            frequency = value["prob"]
            probability = frequency / frequencies[u]

            #graph
            value["prob"] = probability
            graph[u][actionIndex] = value

            #reverse graph
            v = value["obs"]
            #value["obs"] = stateIndex
            
            if v not in reverseGraph.keys():
                reverseGraph[v] = {}
            
            value = {}
            value["obs"] = u
            value["action"] = actionIndex
            prevVal = reverseGraph[v]
            idx = len(prevVal)
            prevVal[idx] = value
            reverseGraph[v] = prevVal

        if u not in reverseGraph.keys():
            reverseGraph[u] = {}


    for s in reverseGraph.keys():
        if s not in graph.keys():
            graph[s] = {}

    graph["gooooooooooooal"] = {}
    print("\n\n\nGRAPH--->\n\n" + str(graph))
    #raise Exception("graph")
    print("\nREVERSE GRAPH--->\n\n" + str(reverseGraph))
    os.remove("graph.txt")
    print(graph, file=open("graph.txt", "w"))


    sequences = buildSequences(graph, reverseGraph, 0.02, 0.02, 10, args.model, statesInfo)
    if args.model == "Model_A" or args.model == "Model_B" or args.model == "Model_C":
        #construct and visualize the intentional sequences
        gifFilename =  "Results/intentional/" + str(args.model) + "/" + str(args.env) + "_" + str(args.seed) + "_"
        visualize(sequences, images, envMap, args.env, args.model, args.argmax, args.seed, args.memory, "", args.episodes, 1, gifFilename, model_dir, agent, statesInfo)

    elif args.model == "Model_D" or args.model == "Model_E" or args.model == "Model_F":
        #construct and visualize the random sequences
        randomSequences = buildRandomSequences(graph, reverseGraph, sequences, args.model, statesInfo)
        rdmFilename = "Results/random/" + str(args.model) + "/" + str(args.env) + "_" + str(args.seed) + "_Random"
        visualize(randomSequences, images, envMap, args.env, args.model, args.argmax, args.seed, args.memory, "", args.episodes, 1, rdmFilename, model_dir, agent, statesInfo)
    else:
        "Please enter a valid model. Ex) Model_A, Model_B, ..., Model_F"

if __name__ == '__main__':
    main()