import argparse
import csv
import heapq as hq
import math
from numpy.linalg import norm 
import pandas as pd
import utils
import torch
import os
from torch_ac.utils.penv import ParallelEnv
from scripts.intentions import getStateIntentionality, getImportance, getConfidence, is_integer


def getIntentionalStates(graph):
    states = sorted(graph)
    # print("states:"+str(states))
    intentStates = {}
    for stateIndex in states:
        probabilities = []
        rewards = []
        for action in graph[stateIndex].keys():
            value = graph[stateIndex][action]
            rewards.append(value["reward"])
            probabilities.append(value["prob"])
        
        intentStates[stateIndex] = getStateIntentionality(rewards, probabilities)
        # print("rewards:"+str(rewards))
        # print("probabilities:"+str(probabilities))
        # print("intent:"+str(intentStates[stateIndex]))

    return intentStates


def getSequence(state, MDP, intentStates, reverse, threshold):
    queue = [state]
    visited = {}
    sequence = []
    count = 0

    while len(queue) > 0:
        currState = queue.pop(0)
        if currState in MDP.keys():
            max = -1
            maxState = -1

            visited[currState] = True 
            
            for actionIndex in MDP[currState].keys():
                values = MDP[currState][actionIndex]
                state = values["obs"]
                
                if state not in intentStates.keys():
                    intentStates[state] = 0
                
                if intentStates[state] > max and state not in visited.keys() and (intentStates[state] > threshold or len(MDP[currState].keys() == 1)):
                    max = intentStates[state]
                    maxState = state
                    maxAction = actionIndex

            if maxState not in visited.keys() and maxState != -1:
                queue.append(maxState)

               # if count == 0: 
                 #   visited[currState] = True                   
                if reverse:
                    sequence.insert(0, "a" + str(maxAction))
                    sequence.insert(0, "s" + str(maxState))
                else:
                    sequence.append("a" + str(maxAction))
                    sequence.append("s" + str(maxState))

                count = count + 1
                
                
                """
                if reverse:
                    sequence.insert(0, "a" + str(maxAction))
                    sequence.insert(0, "s" + str(maxState))
                else:
                    sequence.append("a" +  str(maxAction))
                    sequence.append("s" + str(maxState))
                """

                visited[maxState] = True #visited[currState] = True
        else:
            continue
        
    return sequence
    

def buildSequences(graph, reverseGraph, threshold, budget):

    print("building sequences...")
    def getSequenceIntentionality(sequence):
        intentionality = 1
        for i in range(len(sequence)):
            if i % 2 == 0:
                state = sequence[i]
                state_id = state[1:]
                intentionality = intentionality * intentStates[state_id]
        return intentionality

    sequences = {}
    count = 0

    # get the data for the algorithm
    MDP = graph
    rMDP = reverseGraph
    intentStates = getIntentionalStates(graph)
    # print("intentionalstates:"+str(intentStates))

     # Sort a tuple based on value 'v'
    def getKey(item):
        return item[1]

    # Sort the states by their intentionality
    intentTuple = [(k, v) for k, v in intentStates.items()]
    # print("intentTuple:"+str(intentTuple))
    intentTuple = sorted(intentTuple, key=getKey, reverse=True)
    intents = dict(intentTuple)

    intentionalities = {}
    os.remove("intentionality.txt")

    # Parse through each state from highest to lowest intent, extract the intentional sequences
    for key in intents.keys():
        val = {}
        if count < budget:
            state = key
            context = getSequence(state, rMDP, intentStates, True, threshold)
            future = getSequence(state, MDP, intentStates, False, threshold)
            sequence = context + ["s" + str(state)] + future
            val["sequence"] = sequence
            # print("sequence:"+str(sequence))
            val["intentionality"] = getSequenceIntentionality(sequence)
            #print("intent-> "+ str(val["intentionality"]))
            
            intentionalities[count] = val["intentionality"]
            sequences[count] = val
            count = count + 1
        else:
            break
    
    for intent in intentionalities.keys():
        print(str(intent) + ": " + str(intentionalities[intent]), file=open("intentionality.txt", "a"))
    
    # print(str(sequences))
    return sequences