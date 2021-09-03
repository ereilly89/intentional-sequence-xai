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
        keys = obss
        #print("obss:"+str(obss))
        actions = agent.get_actions(obss)

        startEnvStateID = env.envs[0].hash()
        print("startEnvStateID = " + startEnvStateID)

        obss, rewards, dones, _ = env.step(actions)

        print("actionID = " + str(actions[0]))

        endEnvStateID = env.envs[0].hash()
        print("endEnvStateID: " + endEnvStateID)
        
        # print("observations:"+str(obss))

        for i in range(args.procs):
            testKey = keys[i]
            #print("keys[i]"+str(keys[i]))

            print("\n"+str(i)+"\n")

            stateIndex = startEnvStateID #getHash(keys[i])
            endIndex = endEnvStateID #getHash(obss[i])

            actionIndex = actions[i]
            value = {"obs":    endIndex,
                     "reward": rewards[i]}
           
            if stateIndex not in images.keys():
                images[stateIndex] = keys[i] 

            if stateIndex not in envMap.keys():
                envMap[stateIndex] = copy.deepcopy(env.envs[i])  # print(print("copy:"+str(copy.deepcopy(env.unwrapped.clone_full_state())))) #copy.deepcopy(env)
                #print("deep copy!!!"+ str(envMap[stateIndex]))

            if stateIndex not in statesInfo.keys():

                #get the is_locked and is_open attributes of Door
                #and set them in the info dict below
                door_x = env.envs[i].door_pos[0]
                door_y = env.envs[i].door_pos[1]
                door = env.envs[i].grid.get(door_x, door_y)

                print("door_x: " + str(door_x))
                print("door_y" + str(door_y))
                print("door:" + str(door))
                print("door.is_locked" + str(door.is_locked))
                print("door.is_open" + str(door.is_open))

                info = {
                    "grid": env.envs[i].grid,
                    "grid_encoding": env.envs[i].grid.encode().tolist(),
                    "envGrid": env.envs[i].grid.copy(),
                    "splitIdx": env.envs[i].splitIdx,
                    "doorIdx": env.envs[i].doorIdx,
                    "door_pos": env.envs[i].door_pos,
                    "key_pos": env.envs[i].key_pos,
                    "agent_pos": env.envs[i].agent_pos,
                    "agent_dir": env.envs[i].agent_dir,
                    "carrying": env.envs[i].carrying,
                    "step_count": env.envs[i].step_count,
                    "is_locked": door.is_locked,
                    "is_open": door.is_open
                }
                statesInfo[stateIndex] = info

            # graph
            if stateIndex in graph.keys():
               
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
           
            # get frequencies
            if stateIndex in frequencies.keys():
                freq = frequencies[stateIndex]
                frequencies[stateIndex] = freq + 1
            else:
                frequencies[stateIndex] = 1

            break #temporary? no

        count = count + 1   
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


    

    # convert frequencies to probabilities
    reverseGraph = {}
    for stateIndex in graph.keys():
        for actionIndex in graph[stateIndex].keys():
            #print("edges:"+str(edges))
            value = graph[stateIndex][actionIndex]

            frequency = value["prob"]
            probability = frequency / frequencies[stateIndex]

            #graph
            value["prob"] = probability
            graph[stateIndex][actionIndex] = value

            #reverse graph
            tempStateIndex = value["obs"]
            value["obs"] = stateIndex
            
            if tempStateIndex not in reverseGraph.keys():
                reverseGraph[tempStateIndex] = {}
            
            reverseGraph[tempStateIndex][actionIndex] = value


    print("\n\n\nGRAPH--->\n\n" + str(graph))
    os.remove("graph.txt")
    print(graph, file=open("graph.txt", "w"))

    #construct and visualize the intentional sequences
    sequences = buildSequences(graph, reverseGraph, 0, 10, args.model)
    gifFilename =  "Results/intentional/" + str(args.model) + "/" + str(args.env) + "_" + str(args.seed) + "_"
    visualize(sequences, images, envMap, args.env, args.model, args.argmax, args.seed, args.memory, "", args.episodes, 1, gifFilename, model_dir, agent, statesInfo)

    #construct and visualize the random sequences
    randomSequences = buildRandomSequences(graph, reverseGraph, sequences, args.model)
    rdmFilename = "Results/random/" + str(args.model) + "/" + str(args.env) + "_" + str(args.seed) + "_Random"
    visualize(randomSequences, images, envMap, args.env, args.model, args.argmax, args.seed, args.memory, "", args.episodes, 1, rdmFilename, model_dir, agent, statesInfo)

    end_time = time.time()

    # Print logs
    """
    num_frames = sum(logs["num_frames_per_episode"])
    fps = num_frames/(end_time - start_time)
    duration = int(end_time - start_time)
    return_per_episode = utils.synthesize(logs["return_per_episode"])
    num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

    print("F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
        .format(num_frames, fps, duration,
                *return_per_episode.values(),
                *num_frames_per_episode.values()))
    """

    # Print worst episodes
    """
    n = args.worst_episodes_to_show
    if n > 0:
        print("\n{} worst episodes:".format(n))

        indexes = sorted(range(len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
        for i in indexes[:n]:
            print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))
    """

if __name__ == '__main__':
    main()