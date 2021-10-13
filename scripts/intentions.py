import argparse
import csv
import heapq as hq
import math
from numpy.linalg import norm 
import pandas as pd
#import utils
import torch
from torch_ac.utils.penv import ParallelEnv

# Returns a value between 0 and 1
def getDesire(rwrd):
    n = len(rwrd) 
    imp = 0
    if n == 1:
        return 0

    theMin = math.inf
    #print("REWARD: " + str(rwrd))
    for r in rwrd:
        if float(r) < theMin:
            theMin = float(r)
        
    for r in range(len(rwrd)):
        rwrd[r] = rwrd[r] - theMin

    for r in rwrd:
        if norm(rwrd, 1) == 0:
            return 0
        rnorm = r/norm(rwrd, 1)
        # print("nnorm:"+str(rnorm))
        # print("n:"+str(n))
        try:
            imp += math.log(rnorm, n) * rnorm
        except ValueError as e:
            continue
    imp += 1

    if imp < 0:
        raise Exception("something went wrong.")
        return 0

    return imp


# Returns a value between 0 and 1
def getBelief(prob):
    n = len(prob)
    conf = 0
    if n == 1:
        return 0

    #print("prob:"+str(prob))
    for p in prob:
        conf += math.log(p, n) * p
        #print("conf:"+str(conf))
    conf += 1

    if conf < 0:
        #raise Exception("something went wrong. conf:"+str(conf))
        return 0

    return conf


def getIntentionality(rewards, probabilities):
    
    if len(rewards) == 0:
        return None

    if len(rewards) == 1:
        return 1

    return (getDesire(rewards) + 1) * (getBelief(probabilities) + 1)


def getIntent(state, MDP, normalized):
    rewards = []
    probs = []
    #print("MDP:"+str(MDP))
    #print("MDP[state]:"+str(MDP[state]))
    for actionIndex in MDP[state].keys():
        values = MDP[state][actionIndex]
        rewards.append(values["reward"])
        probs.append(values["prob"])
        #print("\n\n\nrewards: " +str(rewards))
        #print("probs: " + str(probs)+"\n\n\n")

    if getIntentionality(rewards, probs) == None:
        return 1
    
    if normalized:
        if len(probs) == 1:
            return 1
        return (getIntentionality(rewards, probs) - 1)/3
    else:
        return getIntentionality(rewards, probs)



def is_integer(n):
    try:
        int(n)
        return True
    except ValueError:
        return False
