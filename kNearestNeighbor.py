'''
For every point in our dataset:
    - calculate the distance between X and the current point(Eucleadian distance)
    - sort the distances in increasing order
    - take k items with lowest distances to X
    - find the majority class among these items
    - return the majority class as our prediction for the class of X
'''
import operator
import numpy as np

class knnClassifier:
    def __init__(self, X_train, y_train):
        self.xtrain = X_train
        self.labels = y_train


    def classify(self, inputX, k):
        '''
        inputX: input vector(what is to be predicted)
        dataSet: full matrix with training data
        labels: labels vector
        k: the number of nearest neighbours to use in voting
        '''

        dataSetSize = self.xtrain.shape[0]
        
        diffMat = np.tile(inputX, (dataSetSize,1)) - self.xtrain
        # np.tile(inputX, (dataSetSize,1)) repeats the input data so that the size of the input vector is the sanme as that of train data(to allow subtraction)
        
        sqDiffMat = diffMat**2

        sqDistances = sqDiffMat.sum(axis=1)  # the square of differences are summed row-wise(for all the different features)

        distances = sqDistances**0.5    # the square root of the sum is taken to get the distance to each existing data point
        print(distances)

        # sorting the distances
        sortedDistIndicies = distances.argsort()
        print("Sorted Indices: ", sortedDistIndicies)

        classCount={}
        for i in range(k):
            voteIlabel = self.labels[sortedDistIndicies[i]]
            classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        sortedClassCount = sorted(classCount.items(),
                                key=operator.itemgetter(1), reverse=True)

        return sortedClassCount[0][0]


myKNN = knnClassifier(X_train=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]]), y_train=['A','A','B','B'])

print(myKNN.classify([0,0],  3))
