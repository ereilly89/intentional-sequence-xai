import argparse
import csv
import heapq as hq
import math
from numpy.linalg import norm 
import pandas as pd
import utils
import torch
from torch_ac.utils.penv import ParallelEnv

# Returns a value between 0 and 1
def getImportance(rwrd):
    n = len(rwrd) 
    imp = 0
    if n == 1:
        return 0

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
        return 0

    return imp


# Returns a value between 0 and 1
def getConfidence(prob):
    n = len(prob)
    conf = 0
    if n == 1:
        return 0

    for p in prob:
        conf += math.log(p, n) * p
    conf += 1

    if conf < 0:
        return 0

    return conf


def getStateIntentionality(rewards, probabilities):
    return (getImportance(rewards) + 1) * (getConfidence(probabilities) + 1)


def is_integer(n):
    try:
        int(n)
        return True
    except ValueError:
        return False
