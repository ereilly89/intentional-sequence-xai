import pandas as pd


def computeCorr():
    intentCorrDF = pd.read_csv("intentional_correlation.csv")
    randomCorrDF = pd.read_csv("random_correlation.csv")

    userRank = intentCorrDF.loc[:,"userRank"]
    actualRank = intentCorrDF.loc[:,"actualRank"]
    print("Pearson Correlation: " + str(userRank.corr(actualRank)))
    print("Spearman Correlation: " + str(userRank.corr(actualRank, method='spearman')))
    
    randomUserRank = randomCorrDF.loc[:,"userRank"]
    randomActualRank = randomCorrDF.loc[:,"actualRank"]
    print("Pearson Correlation: " + str(userRank.corr(actualRank)))
    print("Spearman Correlation: " + str(userRank.corr(actualRank, method='spearman')))


def computeAvgDiff():
    intentEval = pd.read_csv("intentional_evaluation.csv")
    randomEval = pd.read_csv("random_evaluation.csv")

    prediction = intentEval.loc[:,"prediction"]
    actual = intentEval.loc[:,"actual"]

    difference = actual - prediction
    sumDifferenceSquared = (difference * difference).sum()
    avgDifferenceSquared = sumDifferenceSquared / len(prediction)

    print("avgDifferenceSquared: " + str(avgDifferenceSquared))


def main():
    computeCorr()
    computeAvgDiff()


main()