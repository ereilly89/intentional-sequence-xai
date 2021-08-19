import csv
from scripts.intentions import getStateIntentionality, getImportance, getConfidence, is_integer

def getIntentionalStates(dtm_output):
    probabilities = []
    rewards = []
    prevState = -1

    with open(dtm_output, 'r') as csvfile:
       datareader = csv.reader(csvfile)
       intentStates = {}
       for row in datareader:
            if is_integer(row[0]):
                state = int(row[0])
                if state != prevState and len(probabilities) > 0:
                    intentStates[prevState] = getStateIntentionality(rewards, probabilities)
                    probabilities.clear()
                    rewards.clear()
                probabilities.append(float(row[3]))
                rewards.append(float(row[4]))
                prevState = int(row[0])
            elif row[0] == "Start state index":
                continue
            else:
                break
    
    return intentStates


def getGraph(dtm_output, reverse):
    graph = {}
    with open(dtm_output, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        values = {}
        val = {}
        i = 0
        prevState = -1
        for row in datareader:
            
            if is_integer(row[0]):

                if not reverse and prevState != row[0] and len(values) > 0:
                    graph[row[0]] = values
                    values = {}
                    i = 0
                elif reverse and prevState != row[2] and len(values) > 0:
                    graph[row[2]] = values
                    values = {}
                    i = 0
                elif prevState == row[0]:
                    i = i + 1

                val["action"] = row[1]
                val["probability"] = row[3]
                val["reward"] = row[4]

                if not reverse:
                    val["state_id"] = row[2]
                    values[i] = val
                    val = {}
                    prevState = row[0]
                else:
                    val["state_id"] = row[0]
                    values[i] = val
                    val = {}
                    prevState = row[2]

            elif row[0] == "Start state index":
                continue

            else:
                break

    return graph


def getSequence(state, MDP, intentStates, reverse, threshold):
    queue = [state]
    visited = {}
    sequence = []

    while len(queue) > 0:
        curr = queue.pop(0)
        if str(curr) in MDP.keys():
            values = MDP[str(curr)]
            max = -1
            maxState = -1
            for i in values:
                # {'action': _, 'probability': _, 'reward': _, 'state_id': _}
                tuple = values[i]
                state = tuple["state_id"]
                action = tuple["action"]

                
                if int(state) not in intentStates.keys():
                    intentStates[int(state)] = 0

                if intentStates[int(state)] > max and intentStates[int(state)] > threshold:
                    max = intentStates[int(state)]
                    maxState = state
                    maxAction = action

            if maxState not in visited.keys() and maxState != -1:
                queue.append(maxState)
                if reverse:
                    sequence.insert(0, "a" + str(maxAction))
                    sequence.insert(0, "s" + str(maxState))
                else:
                    sequence.append("a" +  str(maxAction))
                    sequence.append("s" + str(maxState))

            visited[curr] = True
        else:
            continue
        
    return sequence
    

def buildSequences(dtm_output, threshold, budget):


    def getSequenceIntentionality(sequence):
        intentionality = 1
        for i in range(len(sequence)):
            if i % 2 == 0:
                state = sequence[i]
                state_id = int(state[1:])
                intentionality = intentionality * intentStates[state_id]
        return intentionality

    sequences = {}
    count = 0

    # get the data for the algorithm
    MDP = getGraph(dtm_output, False)
    rMDP = getGraph(dtm_output, True)
    intentStates = getIntentionalStates(dtm_output)

     # Sort a tuple based on value 'v'
    def getKey(item):
        return item[1]

    # Sort the states by their intentionality
    intentTuple = [(k, v) for k, v in intentStates.items()]
    intentTuple = sorted(intentTuple, key=getKey, reverse=True)
    intents = dict(intentTuple)

    # Parse through each state from highest to lowest intent, extract the intentional sequences
    for key in intents.keys():
        val = {}
        if count < budget:
            state = key
            context = getSequence(state, rMDP, intentStates, True, threshold)
            future = getSequence(state, MDP, intentStates, False, threshold)
            sequence = context + ["s" + str(state)] + future
            val["sequence"] = sequence
            val["intentionality"] = getSequenceIntentionality(sequence)
            sequences[count] = val
            count = count + 1
        else:
            break
    #print(str(sequences))


def main():
    # specify the dtm output file to be parsed
    dtm_output = "dtm-output.csv"
    sequences = buildSequences(dtm_output, threshold=0, budget=10) #threshold must be greater than or equal to 0
    
#main()