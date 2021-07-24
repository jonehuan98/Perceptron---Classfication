import numpy as np
import pandas as pd

class Perceptron:
    def __init__(self, alpha=0.01, iterations=20):
        self.alpha = alpha
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def unitStepFunction(self, value):
        return np.where(value>=0, 1, 0) #returns 1 if x>=0, otherwise return 0

    def linearOutput(self, features):
        score = np.dot(features, self.weights) + self.bias
        return score
    
    def fitting(self, X, Y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = Y
        for _ in range(self.iterations):
            for idx, x_i in enumerate(X): #idx is index of sample, x_i is value of sample
                #calculate predicted value
                y_predicted = self.unitStepFunction(self.linearOutput(x_i))
                #put into update formula
                update = self.alpha * (y_[idx] - y_predicted)
                self.weights += (update * x_i)
                self.bias += update

    def regularizationFitting(self, X, Y, lam):
        self.lam = lam
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)        
        self.bias = 0
        y_ = Y
        for _ in range(self.iterations):
            for idx, x_i in enumerate(X): #idx is index of sample, x_i is value of sample
                y_predicted = self.unitStepFunction(self.linearOutput(x_i))
                update = self.alpha * (y_[idx] - y_predicted)
                self.weights = (1-(2*self.lam))*self.weights + (update * x_i)
                self.bias += update

    def predict(self, X):
        y_predicted = self.unitStepFunction(self.linearOutput(X))
        return y_predicted

def main():
    
    trainData = np.array(pd.read_csv("train.data", header = None))
    testData = np.array(pd.read_csv("test.data", header = None))

    #splitdata
    trainOne = trainData[0:40] 
    trainTwo = trainData[40:80]
    trainThree = trainData[80:120]
    testOne = testData[0:10]
    testTwo = testData[10:20]
    testThree = testData[20:30]
    np.random.seed(100)

    def accuracy(predicted, target):
        accuracy = np.sum(predicted == target)/ len(target)
        return accuracy

    def formatdata(className_1, className_2):
        classOneTarget = className_1
        classOneTarget[:,4] =1
        classTwoTarget = className_2
        classTwoTarget[:,4] =0
        batchData = np.vstack((classOneTarget,classTwoTarget))
        np.random.shuffle(batchData)
        featureList = np.array((batchData[:,:4])).astype(np.float)
        targetList = np.array((batchData[:,4])).astype(np.float)
        return featureList, targetList
    
    def multiclassFormat(className_1, className_2, className_3):
        classOneTarget = className_1
        classOneTarget[:,4] =1
        classTwoThreeTarget = np.vstack((className_2,className_3))
        classTwoThreeTarget[:,4] =0
        batchData = np.vstack((classOneTarget,classTwoThreeTarget))
        np.random.shuffle(batchData)
        featureList = np.array((batchData[:,:4])).astype(np.float)
        targetList = np.array((batchData[:,4])).astype(np.float)
        return featureList, targetList    
    
    def multiclassTesting():
        #check probability using test data
        print("Testing class 1 probability:")
        probability(testOne, multiOne, multiTwo, multiThree)
        print("Testing class 2 probability:")
        probability(testTwo, multiOne, multiTwo, multiThree)
        print("Testing class 3 probability:")
        probability(testThree, multiOne, multiTwo, multiThree)
        print("Training class 1 probability:")
        probability(trainOne, multiOne, multiTwo, multiThree)
        print("Training class 2 probability:")
        probability(trainTwo, multiOne, multiTwo, multiThree)
        print("Training class 3 probability:")
        probability(trainThree, multiOne, multiTwo, multiThree)

    def probability(rawTestData, class1, class2, class3):
        inputProbability = rawTestData[:,:4]
        predictedClass1 = []
        predictedClass2 =[]
        predictedClass3 = []
        for i in inputProbability:
            sampleScore = []
            predictionsOne = class1.linearOutput(i)
            predictionsTwo = class2.linearOutput(i)
            predictionsThree = class3.linearOutput(i)
            sampleScore.append(predictionsOne)
            sampleScore.append(predictionsTwo)
            sampleScore.append(predictionsThree)
            predictedClass = np.argmax(sampleScore) + 1
            if predictedClass == 1:
                predictedClass1.append(predictedClass)
            elif predictedClass == 2:
                predictedClass2.append(predictedClass)
            else:
                predictedClass3.append(predictedClass)

        print("Probability of identifying as class 1:", len(predictedClass1)/len(inputProbability))
        print("Probability of identifying as class 2:", len(predictedClass2)/len(inputProbability))
        print("Probability of identifying as class 3:", len(predictedClass3)/len(inputProbability))

    print("Binary Classification Q3: ")
    #class one and class two    
    inputs, targets = (formatdata(trainOne, trainTwo))
    onetwo = Perceptron()
    onetwo.fitting(inputs, targets)
    print("Weights: ", onetwo.weights)
    predictions = onetwo.predict(inputs)
    print("Train data 1 & 2 accuracy: " , accuracy(predictions, targets))
    inputs, targets = (formatdata(testOne, testTwo))
    predictions = onetwo.predict(inputs)
    print("Test data 1 & 2 accuracy: " , accuracy(predictions, targets))

    #class two and class three
    inputs, targets = (formatdata(trainTwo, trainThree))
    twothree = Perceptron()
    twothree.fitting(inputs, targets)
    print("Weights:" , twothree.weights)
    predictions = twothree.predict(inputs)
    print("Train data 2 & 3 accuracy: " , accuracy(predictions, targets))   
    inputs, targets = (formatdata(testTwo, testThree))
    predictions = twothree.predict(inputs)
    print("Test data 2 & 3 accuracy: " , accuracy(predictions, targets))    

    #class one and class three
    inputs, targets = (formatdata(trainOne, trainThree))
    onethree = Perceptron()
    onethree.fitting(inputs, targets)
    print("Weights:" , onethree.weights)
    predictions = onethree.predict(inputs)
    print("Train data 1 & 3 accuracy: " , accuracy(predictions, targets))
    inputs, targets = (formatdata(testOne, testThree))
    predictions = onethree.predict(inputs)
    print("Test data 1 & 3 accuracy: " , accuracy(predictions, targets))
    
    #Multiclass
    print()
    print("Multiclass classification Q4: ")

    #train multiclass
    #class 1 as 1, rest 0
    inputsTrain1, targetsTrain1 = multiclassFormat(trainOne, trainTwo,trainThree)
    multiOne = Perceptron()
    multiOne.fitting(inputsTrain1, targetsTrain1)
    print("Weights: ", multiOne.weights)

    # #class 2 as 1, rest 0
    inputsTrain2, targetsTrain2 = multiclassFormat(trainTwo, trainOne,trainThree)
    multiTwo = Perceptron()
    multiTwo.fitting(inputsTrain2, targetsTrain2)
    print("Weights: ", multiTwo.weights)

    # #class 3 as 1, rest 0
    inputsTrain3, targetsTrain3 = multiclassFormat(trainThree, trainTwo,trainOne)
    multiThree = Perceptron()
    multiThree.fitting(inputsTrain3, targetsTrain3)
    print("Weights: ", multiThree.weights)

    #test trained perceptrons
    multiclassTesting()

    #Regularisation
    print("Q5:")
    #train and test perceptron with regularisation coefficients
    for i in [0.01, 0.1, 1.0, 10.0, 100.0]:
        print()
        print("Results for lambda ", i)
        multiOne.regularizationFitting(inputsTrain1, targetsTrain1, float(i))
        multiTwo.regularizationFitting(inputsTrain2, targetsTrain2, float(i))
        multiThree.regularizationFitting(inputsTrain3, targetsTrain3, float(i))

        print("Weights: ", multiOne.weights)
        print("Weights: ", multiTwo.weights)
        print("Weights: ", multiThree.weights)

        multiclassTesting()

if __name__ == '__main__':
    main()
