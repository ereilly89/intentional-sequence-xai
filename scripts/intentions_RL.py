import argparse
import csv
import heapq as hq
import math
from numpy.linalg import norm 
import pandas as pd
import utils
import torch
import os
from os.path import exists
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
              
                if reverse:
                    sequence.insert(0, "a" + str(maxAction))
                    sequence.insert(0, "s" + str(maxState))
                else:
                    sequence.append("a" + str(maxAction))
                    sequence.append("s" + str(maxState))

                count = count + 1

                visited[maxState] = True #visited[currState] = True
        else:
            continue
        
    return sequence
    

def getRandomSequence(currState, MDP, reverse, seqLen):
    import random
    queue = [currState]
    visited = {}
    sequence = []
    while len(queue) > 0 and len(sequence) < seqLen:
        currState = queue.pop(0)

        if currState in MDP.keys():
            visited[currState] = True 
            actionIndex = random.choice(list(MDP[currState].keys()))

            values = MDP[currState][actionIndex]
            state = values["obs"]

            if state not in visited.keys():
                queue.append(state)
            
            if reverse:
                sequence.insert(0, "a" + str(actionIndex))
                sequence.insert(0, "s" + str(state))
            else:
                sequence.append("a" + str(actionIndex))
                sequence.append("s" + str(state))
        else:
            print("something went wrong.")

    return sequence


def computeStats(sequences):
    sumSeqLen = 0
    minSeqLen = math.inf
    maxSeqLen = -1
    sumIntent = 0
    minIntent = math.inf
    maxIntent = -1

    stats = {}
    for s in sequences:
        length = int(len(sequences[s]["sequence"])/2) + 1
        intent = sequences[s]["intentionality"]
        
        if length > maxSeqLen:
            maxSeqLen = length
        if length < minSeqLen:
            minSeqLen = length

        if intent > maxIntent:
            maxIntent = intent
        if intent < minIntent:
            minIntent = intent

        sumSeqLen = sumSeqLen + length
        sumIntent = sumIntent + intent
    
    avgSeqLen = sumSeqLen / len(sequences)
    avgIntent = sumIntent / len(sequences)

    stats["avgSeqLen"] = avgSeqLen
    stats["minSeqLen"] = minSeqLen
    stats["maxSeqLen"] = maxSeqLen
    stats["avgIntent"] = avgIntent
    stats["minIntent"] = minIntent
    stats["maxIntent"] = maxIntent

    return stats



def buildSequences(graph, reverseGraph, threshold, budget, model):

    print("building sequences...")

    def getSequenceIntentionality(sequence):
        print("computing sequence intentionality...")
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
    stats = {}
    outputFile = "Results/intentional/" + model + "/intentionality.csv"
    if exists(outputFile):
        os.remove(outputFile)
    
    if model == "Model_A" or model == "Model_B" or model == "Model_C":
        print("Intentionality", file=open(outputFile, "w+"))
    print("parsing through states in order of intentionality...")
    # Parse through each state from highest to lowest intent, extract the intentional sequences
    for key in intents.keys():
        val = {}
        if count < budget:
            state = key

            print("getting context...")
            context = getSequence(state, rMDP, intentStates, True, threshold)
            print("getting future...")
            future = getSequence(state, MDP, intentStates, False, threshold)

            sequence = context + ["s" + str(state)] + future

            val["sequence"] = sequence
            val["intentionality"] = getSequenceIntentionality(sequence)
            intentionalities[count] = val["intentionality"]

            sequences[count] = val
            count = count + 1
        else:
            break
    
    for intent in intentionalities.keys():
        print("model: " + str(model))
        if model == "Model_A" or model == "Model_B" or model == "Model_C":
            print(str(intentionalities[intent]), file=open("Results/intentional/" + model + "/intentionality.csv", "a"))
    
    if model == "Model_A" or model == "Model_B" or model == "Model_C":
        summaryFile = "Results/intentional/" + model + "/summary.csv"
        if exists(summaryFile):
            os.remove(summaryFile)
        summaryDict = computeStats(sequences)
        for key in summaryDict:
            print(str(key) + "," + str(summaryDict[key]), file=open("Results/intentional/" + model + "/summary.csv", "a"))

    return sequences


def buildRandomSequences(graph, reverseGraph, sequences, model):
    # generate random sequences of the same size as the intentional ones
    print("building random sequences...")
    def getSequenceIntentionality(sequence):
        print("computing sequence intentionality...")
        intentionality = 1
        for i in range(len(sequence)):
            if i % 2 == 0:
                state = sequence[i]
                state_id = state[1:]
                intentionality = intentionality * intentStates[state_id]
        return intentionality

    import random
    randomSequences = {}
    intentionalities = {}
    count = 0
    intentStates = getIntentionalStates(graph)
    for s in sequences: 
        sequence = []
        #iters = 0
        while len(sequence) != len(sequences[s]["sequence"]): # or iters > 100:
            startState, endStates = random.choice(list(graph.items()))
            sequence = ["s" + str(startState)]

            print("\n\nrandomly parsing future states...")
            #print("sequences[s]:" + str(sequences[s]["sequence"]))
            #print("len(sequences[s]['sequence']):" + str(len(sequences[s]['sequence'])))
            future = getRandomSequence(startState, graph, False, len(sequences[s]["sequence"])-1)
            context = []
            
            if len(future) != len(sequences[s]["sequence"]) - 1:
                print("randomly parsing context states...")
                context = getRandomSequence(startState, reverseGraph, True, len(sequences[s]["sequence"])-len(future)-1)

            if len(context) > 0:
                sequence = context + sequence
                
            if len(future) > 0:
                sequence = sequence + future

        #if len(sequences[s]["sequence"]) != len(sequence):
         #   print("seq: " + str(sequence))
        print("\n\ns: " + str(s) + ": intentional sequence! : " + str(sequences[s]["sequence"]) + "\nrandom sequence: " + str(sequence))
         #   raise Exception("invalid sequence length")


        val = {}
        val["sequence"] = sequence
        val["intentionality"] = getSequenceIntentionality(sequence)
        outputFile = "Results/random/" + model + "/random_intentionality.csv"

        if exists(outputFile):
            os.remove(outputFile)

        print("Intentionality", file=open(outputFile, "w+"))

        intentionalities[count] = val["intentionality"]
        randomSequences[count] = val

        count = count + 1

        for intent in intentionalities.keys():
            print(str(intentionalities[intent]), file=open("Results/random/" + model + "/random_intentionality.csv", "a"))


        summaryFile = "Results/random/" + model + "/summary.csv"
        if exists(summaryFile):
            os.remove(summaryFile)
        summaryDict = computeStats(sequences)
        for key in summaryDict:
            print(key + "," + str(summaryDict[key]), file=open("Results/random/" + model + "/summary.csv", "a"))


    return randomSequences