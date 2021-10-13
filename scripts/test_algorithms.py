from numpy.linalg import eig
from intentions_RL import getIntentionalContext, getEigenCentralities, getIntentionalPath, getSequenceIntentionality, combineSequences, getMostIntentionalSequence, getIntentionalStates, getBeliefStates, getIntentionalStates, buildSequences, buildRandomSequences
"""
3: {
    "obs": "1",
    "reward": 0,
    "prob": 0.7
}
"""
MDP = {
    "1": {
        0: { 
            "obs": "2",
            "reward": 0,
            "prob": 0.15
        },
        1: { 
            "obs": "3",
            "reward": 1,
            "prob": 0.8
        },
        2: { 
            "obs": "4",
            "reward": 0,
            "prob": 0.05
        },
        
    },
    "2": {
         0: { 
            "obs": "7",
            "reward": 0,
            "prob": 1
        },
    },
    "3": {
        0: { 
            "obs": "8",
            "reward": 10,
            "prob": 0.9
        },
        1: { 
            "obs": "4",
            "reward": -1,
            "prob": 0.1
        },
    },
    "4": {},
    "5": {
        0: { 
            "obs": "1",
            "reward": 0,
            "prob": 0.6
        },
        1: { 
            "obs": "4",
            "reward": 0,
            "prob": 0.4
        },
    },
    "6": {
        0: { 
            "obs": "5",
            "reward": 0,
            "prob": 0.7
        },
        1: { 
            "obs": "7",
            "reward": 0,
            "prob": 0.3
        },
    },
    "7": {
        0: { 
            "obs": "1",
            "reward": 0,
            "prob": 0.5
        },
        1: { 
            "obs": "3",
            "reward": 1,
            "prob": 0.5
        },
    },
    "8": {},
}


#reverse the graph
rMDP = {}
for u in MDP.keys():
    for actionIndex in MDP[u].keys():

        value = MDP[u][actionIndex]
        v = value["obs"]

        if v not in rMDP.keys():
            rMDP[v] = {}
        
        value = {}
        value["obs"] = u
        value["action"] = actionIndex
        prevVal = rMDP[v]
        idx = len(prevVal)
        prevVal[idx] = value
        rMDP[v] = prevVal
    if u not in rMDP.keys():
        rMDP[u] = {}


normIntentStates = getIntentionalStates(MDP, True)
intentStates = getIntentionalStates(MDP, False)
beliefStates = getBeliefStates(MDP)

eigenCentralities = getEigenCentralities(MDP)
print("eigenCentralities: " +str(eigenCentralities))

#sorted()
goals = {}
eigenThreshold = 0.5
for state in eigenCentralities:
    if eigenCentralities[state] > eigenThreshold:
        goals[state] = eigenCentralities[state]

budget = 2
count = 0

for key in beliefStates.keys():
    print("key, " + str(key))
    val = {}
    if count < budget:
        source = key
        print("\nsource: " + str(source))
        #print("goals: " +str(goals))
        intentContext = getIntentionalContext(source, MDP, rMDP, maxContextLength=2)
        intentSeq = getIntentionalPath(key, MDP, goals, eigenCentralities)
        print("intentContext: " +str(intentContext))
        print("intentSeq: " + str(intentSeq))
        count = count + 1

"""
future = getMostIntentionalSequence("1", MDP, rMDP, normIntentStates, beliefStates, False, 0.5, 2)
context = getMostIntentionalSequence("1", MDP, rMDP, normIntentStates, beliefStates, True, 0.003, 2)

intentSeq = combineSequences(context, future)
"""



#print("future: " + str(future))
#print("context: " + str(context))
#print("\nintentSeq: " + str(intentSeq)+"\n")
#print("seq intent: " + str(getSequenceIntentionality(intentSeq, intentStates)))