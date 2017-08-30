# Learn Machine Learning in Action

- The code is for the book of *Machine Learning in Action* study notes
- All of the codes is based on Python2.7
- The source author is pbharrin
- Modify by Neko

## ch02 Classifying with k-Nearest Neighbors
- code segment

```python
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse = True)
    # print sortedClassCount
    return sortedClassCount[0][0]

```
