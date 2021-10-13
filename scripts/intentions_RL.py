import argparse
import csv
import heapq as hq
import math
from numpy.linalg import norm 
import pandas as pd
#import utils
import torch
import os
from os.path import exists
from torch_ac.utils.penv import ParallelEnv
from scripts.intentions import getIntent, getIntentionality, getDesire, getBelief, is_integer


def getIntentionalStates(graph, normalized):
    print("\n\nGETTING INTENTIONAL STATES! \n\n")
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
        
        if getIntentionality(rewards, probabilities) == None:
            #if normalized == True:
            #    intentStates[stateIndex] = 0 #was None
            #else:
            intentStates[stateIndex] = 1 #was None
        else:
            if normalized == True:
                intentStates[stateIndex] = (getIntentionality(rewards, probabilities) - 1) / 3
            else:
                intentStates[stateIndex] = getIntentionality(rewards, probabilities)
        # print("rewards:"+str(rewards))
        # print("probabilities:"+str(probabilities))
        # print("intent:"+str(intentStates[stateIndex]))

    return intentStates

def getBeliefStates(graph):
    def getKey(item):
        return item[1]

    states = sorted(graph)
    beliefStates = {}
    for stateIndex in states:
        probabilities = []
        for action in graph[stateIndex].keys():
            value = graph[stateIndex][action]
            probabilities.append(value["prob"])

        beliefStates[stateIndex] = getBelief(probabilities)

    # Sort the states by their intentionality
    beliefTuple = [(k, v) for k, v in beliefStates.items()]
    beliefTuple = sorted(beliefTuple, key=getKey, reverse=True)
    intents = dict(beliefTuple)

    return beliefStates

def combineSequences(context, future):
    intentSeq = []

    if len(future) > 0 and len(context) > 0:
        if future[0] != context[0]:
            raise Exception("sequence mismatch error.")

    for i in range(len(context)):
        intentSeq.append(context[len(context)-i-1])
    for i in range(len(future)-1):
        intentSeq.append(future[i+1])

    return intentSeq


def getSequenceIntentionality(sequence, intentStates):
    raise Exception("are you actually using this method? Apparently...")
    """
        print("computing sequence intentionality...")
        intentionality = 1
        for i in range(len(sequence)):
            if i % 2 == 0:
                state = sequence[i]
                state_id = state[1:]
                if intentStates[state_id] != None:
                    intentionality = intentionality * intentStates[state_id]
                    print("intentionality->" + str(intentStates[state_id]))
        raise Exception("getSequenceIntentionality")
        return intentionality
    """

def extractPath(goal, source, prev, prevAction):
    print("\nextracting sequence...")
    #print("goal:"+str(goal))
    sequence = []
    curr = goal
    seqElements = {}
    #print("prev:"+str(prev))
    while True:
        #print("curr: " + str(curr))
        if curr not in prev.keys() or curr == source:
            sequence.insert(0, "s" + str(curr))
            #print("test1")
            break
        #print("source:"+str(source))
        #print("goal:"+str(goal))
        #print("prev:"+str(prev))

        """
        if curr in seqElements.keys():
            print("test2")
            if curr == source:
                raise Exception("big oof")
            break 
        """
        sequence.insert(0, "s" + str(curr))
        sequence.insert(0, "a" + str(prevAction[curr]))
        seqElements[curr] = True
        curr = prev[curr]
       

    return sequence


def getGoalStates(eigenCentralities, threshold):
        
    goals = {}
    for state in eigenCentralities:
        if eigenCentralities[state] > threshold:
            goals[state] = eigenCentralities[state]

    return goals


def getEigenCentralities(MDP):
    import networkx as nx

    print("sorting states by eigenvector centrality...")

    G = nx.Graph()
    for s1 in MDP.keys():
        for action in MDP[s1].keys():
            value = MDP[s1][action]
            s2 = value["obs"]
            G.add_edge(s1, s2, weight=value["prob"])

    def getKey(item):
        return item[1]

    centrality = nx.eigenvector_centrality(G, max_iter=10000)

    # Sort the states by their intentionality
    centralityTuple = [(k, v) for k, v in centrality.items()]
    centralityTuple = sorted(centralityTuple, key=getKey, reverse=True)
    centralities = dict(centralityTuple)
   
    return centralities


def getIntentionalities(intentStates):
    print("sorting states by intentionality...")

     # Sort a tuple based on value 'v'
    def getKey(item):
        return item[1]

    # Sort the states by their intentionality
    intentTuple = [(k, v) for k, v in intentStates.items()]
    intentTuple = sorted(intentTuple, key=getKey, reverse=True)
    intents = dict(intentTuple)

    return intents

def getLowValueStates(MDP):
    print("sortings states by lowest q-value...")


def getIntentionalContext(source, MDP, rMDP, maxContextLength):
    intent = {}
    dist = {}
    prev = {}
    prevAction = {}
    unvisited = {}
    maxIntent = -1
    #print("source...:" + str(source))
    for s in rMDP.keys():
        if s != source:
            intent[s] = math.inf*-1
            dist[s] = math.inf*-1
        else:
            intent[s] = getIntent(s, MDP, False)
            #print("firstIntent: " +str(intent[s]))
            #raise Exception("found first intent, s:"+str(s))
            dist[s] = 0

        unvisited[s] = True

    goal = -1
    while len(unvisited) > 0:
        theMax = -math.inf

        # relax the unvisited state of highest intent
        for i in intent.keys():
            if intent[i] >= theMax and i in unvisited.keys():
                theMax = intent[i]
                u = i

        if theMax == math.inf*-1:
            break

        #print("relaxing u:" + str(u) + ", " + str(intent[u]))
        if dist[u] <= maxContextLength and intent[u] > maxIntent:
            goal = u
            maxIntent = intent[u]

        currIntent = theMax
        unvisited.pop(u) 

        # update most intentional paths
        for idx in rMDP[u].keys():
            values = rMDP[u][idx]
            v = values["obs"]
            #print("adding v: " +str(v))

            if getIntent(v, MDP, False) == None:
                alt = intent[u]
            else:
                alt = intent[u] * getIntent(v, MDP, False)

            if alt > intent[v] and dist[u] < maxContextLength and v in unvisited.keys():
                #print("updated!")
                if alt == 0:
                    intent[v] = intent[u]
                else:
                    intent[v] = alt

                dist[v] = dist[u] + 1
                prev[v] = u
                prevAction[v] = values["action"]

    #print("prev: " + str(prev))
    if goal != -1:
        #print("source: " + str(source))
        #print("goal: " + str(goal))
        #print("prev:"+str(prev))
        #if source == "039bf8886fb746ee":
        #    raise Exception("stop")
        return extractPath(goal, source, prev, prevAction)
    else:
        #print("theIntents:" +str(intent))
        return []


def getIntentionalPath(source, MDP, goals, EVCs):
    print("finding intentional path from high-belief state to state of influence...")
    intent = {}
    dist = {}
    prev = {}
    prevAction = {}
    unvisited = {}
    maxEVC = -1

    for s in MDP.keys(): #for each state in the graph
        if s is not source:
            intent[s] = math.inf*-1
            dist[s] = math.inf*-1
        else:
            intent[s] = getIntent(s, MDP, True)
            dist[s] = 0

        unvisited[s] = True

    while len(unvisited) > 0:
        theMax = -math.inf
    
        # relax the unvisited state of highest intent
        for i in intent.keys():
            if intent[i] >= theMax and i in unvisited.keys():
                theMax = intent[i]
                u = i

        if theMax == math.inf*-1:
            #print(str(prev))
            #raise Exception("uh oh")
            break

        #print("goals:"+str(goals))
        #print("u:"+str(u))
        if u in goals.keys():
            goal = u
            break

        if EVCs[u] > maxEVC:
            goal = u
            #print("updated goal: " + str(goal))
            maxEVC = EVCs[u]

        currIntent = theMax
        unvisited.pop(u) 

        # update most intentional paths
        for actionIndex in MDP[u].keys():
            values = MDP[u][actionIndex]
            v = values["obs"]

            if getIntent(v, MDP, True) == None:
                alt = intent[u]
            else:
                alt = intent[u] * getIntent(v, MDP, True)

            if alt > intent[v]:
                if alt == 0:
                    intent[v] = intent[u]
                else:
                    intent[v] = alt

                prev[v] = u
                prevAction[v] = actionIndex
    


    return extractPath(goal, source, prev, prevAction)


def getMostIntentionalSequence(source, MDP, rMDP, intentStates, beliefStates, reverse, threshold, maxContextLength):
    print("meh")
    """
    intent = {}
    dist = {}
    prev = {}
    prevAction = {}
    pq = []
    unvisited = {}
    print("intentStates: " + str(intentStates))
    print("beliefStates: " + str(beliefStates))
    #raise Exception("test")


    for s in MDP.keys(): #for each state in the graph
        if s is not source:
            intent[s] = math.inf*-1
            dist[s] = math.inf*-1
            #prev[s] = None
        else:
            intent[s] = getIntent(s, MDP, True)
            dist[s] = 0
            print("intent[s]:" + str(intent[s]))

        #hq.heappush(pq, (intent[s], s))
        unvisited[s] = True
    print("initialization: " + str(intent))


    contextLength = 0

    maxState = -1
    maxAction = -1
    theMax = -1
    maxCtxIntent = -1

    count = 0
    while len(unvisited) > 0:
        #currIntent, u = hq.heappop(pq)
        
        print("\n\nNEW ITERATION\n\n")

        theMax = -math.inf
        print("intent:" + str(intent))
    
        for i in intent.keys():
            print("i:"+str(i))
            print("intent[i]:" + str(intent[i]))

            if intent[i] >= theMax and i in unvisited.keys():
                print(str(intent[i]) + " is greater than or equal to " + str(theMax))
                theMax = intent[i]
                u = i


        currIntent = theMax
        unvisited.pop(u)      
        #if u == -1:
        #    break
        print("u: " + str(u))
        #print("theIntent: " + str(getIntent(u, MDP)))
        
        #raise Exception("stop")

        if reverse:
            maxState = -1
            if getIntent(u, MDP, True) is not None:
                if getIntent(u, MDP, True) > threshold: # if the agent found the goal, add it to the sequence
                    if dist[u] == maxContextLength:
                        break
                    
        else:

            if getIntent(u, MDP, True) is not None:
                if getIntent(u, MDP, True) > threshold: # if the agent found the goal, add it to the sequence

                    print("INTENT " + str(getIntent(u, MDP, True)) + " > " + str(threshold))
                    max = -1
                    #maxState = -1
                    for actionIndex in MDP[u].keys():
                        values = MDP[u][actionIndex]
                        v = values["obs"]
                        
                        print("\nCHECKING BELIEF STATE GREATER THAN MAX\n")
                        print("beliefStates[v]:" + str(beliefStates[v]))
                        print("max: " +str(max))
                        if beliefStates[v] > max and v != u:
                            max = beliefStates[v]
                            maxState = v
                            maxAction = actionIndex
                            print("maxState is, " + str(maxState))

                    prev[maxState] = u
                    prevAction[maxState] = maxAction

                    try:
                        if getIntent(v, MDP, True) == None:
                            intent[maxState] = intent[u]
                        else:
                            intent[maxState] = intent[u] * getIntent(v, MDP, True)
                    except:
                        #current state is the goal
                        maxState = u

                    break

                elif getIntent(u, MDP, True) > theMax:
                    #theMax = getIntent(u, MDP)
                    #maxState = u

                    max = -1
                    #maxState = -1
                    for actionIndex in MDP[u].keys():
                        values = MDP[u][actionIndex]
                        v = values["obs"]
                        
                        print("\nCHECKING BELIEF STATE GREATER THAN MAX\n")
                        print("beliefStates[v]:" + str(beliefStates[v]))
                        print("max: " +str(max))
                        if beliefStates[v] > max and v != u:
                            max = beliefStates[v]
                            maxState = v
                            maxAction = actionIndex
                            print("maxState is, " + str(maxState))

                    prev[maxState] = u
                    prevAction[maxState] = maxAction

                    if getIntent(v, MDP, True) == None:
                        intent[maxState] = intent[u]
                    else:
                        intent[maxState] = intent[u] * getIntent(v, MDP, True)
                    


        if reverse:
            print("rMDP: " + str(rMDP))
            print("u:"+str(u)+"\n")
            for idx in rMDP[u].keys():
            # print("values: " + str(values))
                values = rMDP[u][idx]
                v = values["obs"]
                
                print("intent[u]:" + str(intent[u]))

                if getIntent(v, MDP, True) == None:
                    alt = intent[u]
                else:
                    alt = intent[u] * getIntent(v, MDP, True)
                print("alt: " + str(alt))
                if alt > intent[v] and dist[u] < maxContextLength:
                    if alt == 0:
                        print("intent of 0 is the previous intent instead. " + str(v) +"\'s intent = " + str(intent[u]) + " (state " + str(u) +"\'s intent)\n")
                        intent[v] = intent[u]
                    else:
                        print("alt:"+str(alt) + " is more than " + str(intent[v])+"\n")
                        intent[v] = alt

                    if alt > maxCtxIntent:
                        maxContextIntent = alt
                        maxState = v
                        maxAction = values["action"]

                    dist[v] = dist[u] + 1
                    prev[v] = u
                    prevAction[v] = values["action"]
        
        else:
            print("MDP:"+str(MDP)+"\n\n")
            print("u:"+str(u)+"\n")
            for actionIndex in MDP[u].keys():
                
                values = MDP[u][actionIndex]
                v = values["obs"]
                print("state " + str(u) + " -> state " + str(v))
                if getIntent(v, MDP, True) == None:
                    alt = intent[u]
                else:
                    alt = intent[u] * getIntent(v, MDP, True)
                
                #print("\nintent[u]:" + str(intent[u]))
                #print("getIntent(): " + str(getIntent(v, MDP)) + "\n")

                if alt > intent[v]:

                    if alt == 0:
                        print("intent of 0 is the previous intent instead. " + str(v) +"\'s intent = " + str(intent[u]) + " (state " + str(u) +"\'s intent)\n")
                        intent[v] = intent[u]
                    else:
                        print("alt:"+str(alt) + " is more than " + str(intent[v])+"\n")
                        intent[v] = alt

                    #intent[v] = alt
                    prev[v] = u
                    prevAction[v] = actionIndex
                    #hq.heappush(pq, (alt, v))
    
        
        print("prev: " + str(prev))
        print("prevAction: " + str(prevAction))
        print("unvisited: " + str(unvisited))
        print("intent: " + str(intent))
        print("dist:" + str(dist))
        print("rMDP: " + str(rMDP))
        #if count == 8:
        #    raise Exception("test")
        count = count + 1

    #print("no goal given a threshold of " + str(threshold))  

    print("intent:"+str(intent))
    print("prev:" + str(prev))
    print("prevAction: " + str(prevAction))
    print("unvisited: " + str(unvisited))

    
    if reverse:
        sequence = []
        print("maxState: " +str(maxState))
        
        curr = maxState
        if curr == -1:
            sequence = ["s"+source]
            print("Couldn't find a goal state with that threshold.")
        else:

            while True:
                sequence.insert(0, "s" + str(curr))
                sequence.insert(0, "a" + str(prevAction[curr]))
                curr = prev[curr]

                if curr not in prev.keys():
                    sequence.insert(0, "s" + str(curr))
                    break
        print("context...")
    else:
        sequence = []
        curr = maxState
        
        print("theMax: " +str(theMax))
        print("curr:"+str(curr))
        seqElements = {}
        while True:
            
            #if curr not in seqElements.keys():
            print("curr:" +str(curr))
            print("prevAction[curr]:"+str(prevAction[curr]))
            sequence.insert(0, "s" + str(curr))
            sequence.insert(0, "a" + str(prevAction[curr]))
            curr = prev[curr]
            seqElements[curr] = True
            if curr not in prev.keys():
                sequence.insert(0, "s" + str(curr))
                break
        print("forward...")
    
    print("seq: " + str(sequence))
        #
        #   values = MDP[state]
        #    for val in values:
    return sequence
    """

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
            #print("\nMDP: " + str(MDP))
            #print("currState:" +str(currState))
            elements = MDP[currState].keys()
            print("currState: " + str(currState))
            print("MDP[currState]: " + str(MDP[currState]))

            if len(elements) > 0:
                #print("\nelements:"+str(elements))
                #print("seqLen:"+str(seqLen))
                print("elements: " + str(list(elements)))
                actionIndex = -1
                while actionIndex < 0 or actionIndex > 6:
                    actionIndex = random.choice(list(elements))
                #print("actionIndex:"+str(actionIndex))
                
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
    #print("The Sequences: " + str(sequences))
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

    if len(sequences) == 0:
        avgSeqLen = 0
        avgIntent = 0
    else:
        avgSeqLen = sumSeqLen / len(sequences)
        avgIntent = sumIntent / len(sequences)

    stats["avgSeqLen"] = avgSeqLen
    stats["minSeqLen"] = minSeqLen
    stats["maxSeqLen"] = maxSeqLen
    stats["avgIntent"] = avgIntent
    stats["minIntent"] = minIntent
    stats["maxIntent"] = maxIntent
    print("Stats: " + str(stats))
    #raise Exception("stats")
    return stats



def buildSequences(graph, reverseGraph, fthreshold, cthreshold, budget, model, statesInfo):

    print("building sequences...")

    def getSequenceIntentionality(sequence):
        print("computing sequence intentionality...")
        intentionality = 1
        prevState = -1
        for i in range(len(sequence)):
            if i % 2 == 0:
                state = sequence[i]
                #print("sequence:" + str(sequence))
                #print("state:"+str(state))
                state_id = state[1:]

                #intentStates["gooooooooooooal"] = 4
                print("state_id: " + str(state_id))

                if state_id in intentStates.keys():
                    intentionality = intentionality * intentStates[state_id]
                else:
                    intentionality = intentionality * intentStates[prevState]

                prevState = state_id

        return intentionality

    sequences = {}
    count = 0

    # get the data for the algorithm
    MDP = graph
    rMDP = reverseGraph
    intentStates = getIntentionalStates(graph, False)
    beliefStates = getBeliefStates(graph)


    #ordering = getIntentionalities(intentStates)
    eigenCentralities = getEigenCentralities(MDP)
    print("eigenCentralities: " + str(eigenCentralities))
    ordering = getBeliefStates(MDP)
    print("beliefStates: " +str(beliefStates))


    goals = getGoalStates(eigenCentralities, 0.02) #was 0.0001 looks good


    intentionalities = {}
    stats = {}
    outputFile = "Results/intentional/" + model + "/intentionality.csv"
    if exists(outputFile):
        os.remove(outputFile)
    
    if model == "Model_A" or model == "Model_B" or model == "Model_C":
        print("Intentionality", file=open(outputFile, "w+"))

   
    # Parse through each state from highest to lowest intent, extract the intentional sequences
    for key in ordering.keys():
        val = {}

        if count < budget:
            source = key
            print("\nkey: " + str(key) + " is the most belief state -> " + str(beliefStates[key]))
            
            #

            print("\n\ncount: " + str(count) + "\nsource: " + str(source))
            future = getIntentionalPath(source, MDP, goals, eigenCentralities)

            #print("\nTHE FUTURE: " +str(future) + "\n")
            context = getIntentionalContext(source, MDP, rMDP, 2)
            
            #print("THE CONTEXT: " + str(context) + "\n")
            sequence = combineSequences(context, future)

            #print("SEQUENCE: " + str(sequence))

            # Trim the sequence after the main reward is obtained
            
            endIndex = None
            for s in range(len(sequence)):
                element = sequence[s]
                if s % 2 == 0:
                    info = statesInfo[element[1:]]
                    the_agent_pos = info["agent_pos"]
                    the_goal_pos = info["goal_pos"]

                    #print("the_agent_pos[0] & the_goal_pos[0]: " + str(the_agent_pos[0]) + ":" + str(the_goal_pos[0]))
                    #print("the_agent_pos[1] & the_goal_pos[1]: " + str(the_agent_pos[1]) + ":" + str(the_goal_pos[1]))
                    #print("info[agent_dir]: " + str(info["agent_dir"]))
                    if s != len(sequence)-1:
                        #print("the action: " + str(sequence[s + 1]))
                        if (the_agent_pos[0] == 3 and the_agent_pos[1] == 2 and info["agent_dir"] == 1 and sequence[s + 1] == "a2") or (the_agent_pos[0] == 2 and the_agent_pos[1] == 3 and info["agent_dir"] == 0 and sequence[s + 1] == "a2"):
                            #print("triggered condition.")
                            endIndex = s + 2 
                            break

            if endIndex is not None:
                sequence = sequence[0:endIndex + 1]
                sequence[endIndex] = "sgooooooooooooal"
            
            print("sequence: " + str(sequence))

            """
            lenBefore = (len(sequence)-1) // 2

            endIndex = None
            for s in range(len(sequence)):
                element = sequence[s]

                #print("s: " + str(s))
                #print("sequence[s]: " + str(sequence[s]))
                if s % 2 == 0 and s != 0:
                    value = MDP[prevState]
                    #print("value: " + str(value))

                    for action in value:
                        #print("action: " + str(action))
                        node = MDP[prevState][action]
                        #print("node: " + str(node))
                        #print("sequence[s]: " + str(sequence[s]))

                        if node["obs"] == element[1:]:
                            if node["reward"] != 0:
                                endIndex = s
                                print("s: " + str(s))
                                print("endIndex: " + str(endIndex))
                                #raise Exception("nonzero reward. endIndex: " + str(endIndex))

                if s % 2 == 0:
                    prevState = element[1:]
            
            if endIndex is not None:
                sequence = sequence[0:endIndex + 1]
                lenAfter = (len(sequence)-1) // 2
                #print("sequence: " + str(sequence))
                print("count: " + str(count))
                if count == 2:
                    print("lenBefore: " + str(lenBefore))
                    print("lenAfter: " + str(lenAfter)+"\n")
                    raise Exception("stop")
            """
            #print()
            #raise Exception("MDP:"+str(MDP))
            """
            for u in MDP.keys():
                for idx in MDP[u].keys():
                    obj = MDP[u][idx]
                    if obj["reward"] != 0:
                        raise Exception("NONZERO REWARD!!!")
            raise Exception("Whoops")
            """

            #print("rMDP:"+str(rMDP))
            #if count == 2:
            #   raise Exception("test")

            #


            """

            print("getting context...")
            #context = getSequence(state, rMDP, intentStates, True, threshold)
            context = getMostIntentionalSequence(state, MDP, rMDP, intentStates, beliefStates, True, cthreshold, 2)

            #print("context: " +str(context))
            #raise Exception("test")
            print("getting future...")
            #future = getSequence(state, MDP, intentStates, beliefStates, False, threshold)
            future = getMostIntentionalSequence(state, MDP, rMDP, intentStates, beliefStates, False, fthreshold, None)

            #sequence = context + ["s" + str(state)] + future
            sequence = combineSequences(context, future)

            print("context, " + str(context))
            print("future, " + str(future))
            print("seq: " + str(sequence))

            """


            val["sequence"] = sequence
            val["intentionality"] = getSequenceIntentionality(sequence)
            intentionalities[count] = val["intentionality"]

            sequences[count] = val
            count = count + 1
        else:
            print("broken")
            break
    
    for intent in intentionalities.keys():
        #print("model: " + str(model))
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


def buildRandomSequences(graph, reverseGraph, sequences, model, statesInfo):
    # generate random sequences of the same size as the intentional ones
    print("building random sequences...")
    def getSequenceIntentionality(sequence):
        print("computing sequence intentionality...")
        intentionality = 1
        prevState = -1
        for i in range(len(sequence)):
            if i % 2 == 0:
                state = sequence[i]
                state_id = state[1:]

                if state_id in intentStates.keys():
                    intentionality = intentionality * intentStates[state_id]
                else:
                    intentionality = intentionality * intentStates[prevState]

                prevState = state_id

        return intentionality

    import random
    randomSequences = {}
    intentionalities = {}
    count = 0
    intentStates = getIntentionalStates(graph, False)
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


            # Trim the sequence after the main reward is obtained
            """
            endIndex = None
            for st in range(len(sequence)):
                element = sequence[st]
                if st % 2 == 0:
                    info = statesInfo[element[1:]]
                    the_agent_pos = info["agent_pos"]
                    the_goal_pos = info["goal_pos"]

                    #print("the_agent_pos[0] & the_goal_pos[0]: " + str(the_agent_pos[0]) + ":" + str(the_goal_pos[0]))
                    #print("the_agent_pos[1] & the_goal_pos[1]: " + str(the_agent_pos[1]) + ":" + str(the_goal_pos[1]))
                    #print("info[agent_dir]: " + str(info["agent_dir"]))
                    if st != len(sequence)-1:
                        #print("the action: " + str(sequence[s + 1]))
                        if (the_agent_pos[0] == 3 and the_agent_pos[1] == 2 and info["agent_dir"] == 1 and sequence[st + 1] == "a2") or (the_agent_pos[0] == 2 and the_agent_pos[1] == 3 and info["agent_dir"] == 0 and sequence[st + 1] == "a2"):
                            #print("triggered condition.")
                            endIndex = st + 2 
                            break

            if endIndex is not None:
                sequence = sequence[0:endIndex + 1]
                sequence[endIndex] = "sgooooooooooooal"
            """
            print("sequence: " + str(sequence))
            



        #if len(sequences[s]["sequence"]) != len(sequence):
         #   print("seq: " + str(sequence))
        #print("\n\ns: " + str(s) + ": intentional sequence! : " + str(sequences[s]["sequence"]) + "\nrandom sequence: " + str(sequence))
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
        summaryDict = computeStats(randomSequences)
        for key in summaryDict:
            print(key + "," + str(summaryDict[key]), file=open("Results/random/" + model + "/summary.csv", "a"))


    return randomSequences