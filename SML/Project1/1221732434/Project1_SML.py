import numpy as np
from scipy import stats
import scipy
import math
import geneNewData
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_context('notebook')
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import metrics




#task1 calculate mean and std
trainData1feature1=[]
trainData1feature2=[]
trainData2feature1=[]
trainData2feature2=[]

testData1feature1=[]
testData1feature2=[]
testData2feature1=[]
testData2feature2=[]


def calculateProbabilityMatrix(data, norm1,norm2, norm3,norm4,testData1feature1,testData2feature1):
    predictionMatrix = np.empty((len(data), 2))
    for i in range(len(data)):
        prob_feature1_digit0 = norm1.pdf(
            testData1feature1[i])
        prob_feature2_digit0 = norm2.pdf(
            testData2feature1[i])
        prob_feature1_digit1 = norm3.pdf(
            testData1feature1[i])
        prob_feature2_digi1 = norm4.pdf(
            testData2feature1[i])

        BaysenDigit0 = prob_feature1_digit0 * \
            prob_feature2_digit0*0.5
        BaysenDigit1 = prob_feature1_digit1 * \
            prob_feature2_digi1*0.5
        predictionMatrix[i, 0] = BaysenDigit0
        predictionMatrix[i, 1] = BaysenDigit1
        final_result = np.argmax(predictionMatrix, axis=1)
    return final_result

#for task 1
def calculateMean(meanTestData,data):
    for i in data:
        mean_brightness = np.mean(i)
        meanTestData.append(mean_brightness)
    return meanTestData

def calculateSTD(stdTestData,data):
    for i in data:
        stdBrighness = np.std(i)
        stdTestData.append(stdBrighness)
    print("Length of STD Data",len(stdTestData))
    return stdTestData

#for task 2
def calculate_digit0():
    meanFeature1Train0 = np.mean(trainData1feature1)
    varianceFeature1Train0 = np.var(trainData1feature1)
    meanFeature2Train0 = np.mean(trainData1feature2)
    varianceFeature2Train0 = np.var(trainData1feature2)
    
    return meanFeature1Train0, varianceFeature1Train0, meanFeature2Train0, varianceFeature2Train0

def calculate_digit1():
    meanFeature1Train1 = np.mean(trainData2feature1)
    varianceFeature1Train1 = np.var(trainData2feature1)
    meanFeature2Train1 = np.mean(trainData2feature2)
    varianceFeature2Train1 = np.var(trainData2feature2)
    return meanFeature1Train1, varianceFeature1Train1, meanFeature2Train1, varianceFeature2Train1


def main():
    myID='2434'
    geneNewData.geneData(myID)
    Numpyfile0 = scipy.io.loadmat('digit0_stu_train'+myID+'.mat')
    Numpyfile1 = scipy.io.loadmat('digit1_stu_train'+myID+'.mat')
    Numpyfile2 = scipy.io.loadmat('digit0_testset'+'.mat')
    Numpyfile3 = scipy.io.loadmat('digit1_testset'+'.mat')
    train0 = Numpyfile0.get('target_img')
    train1 = Numpyfile1.get('target_img')
    test0 = Numpyfile2.get('target_img')
    test1 = Numpyfile3.get('target_img')

    
    # for training: 
    Mean_of_train0= calculateMean(trainData1feature1,train0)
    sns.distplot(Mean_of_train0, fit = stats.norm ,kde=False)
    
    std_of_train0 = calculateSTD(trainData1feature2,train0)
    sns.distplot(std_of_train0, fit = stats.norm ,kde=False)
    

    Mean_of_train1 = calculateMean(trainData2feature1,train1)
    sns.distplot(Mean_of_train1, fit = stats.norm ,kde=False)

    std_of_train1 = calculateSTD(trainData2feature2,train1)
    sns.distplot(std_of_train1, fit = stats.norm ,kde=False)
    
    # for testing 
    plt.figure ()
    Mean_of_test0= calculateMean(testData1feature1,test0)
    sns.distplot(Mean_of_test0, fit = stats.norm ,kde=False)

    std_of_test0= calculateSTD(testData2feature1,test0)
    sns.distplot(std_of_test0, fit = stats.norm ,kde=False)

    
    Mean_of_test1 = calculateMean(testData1feature2,test1)
    sns.distplot(Mean_of_test1, fit = stats.norm ,kde=False)

    std_of_test1 = calculateSTD(testData2feature2,test1)
    sns.distplot(std_of_test1, fit = stats.norm ,kde=False)

    
    #------------Task 2-----------
            
    plt.figure ()
    meanFeature1Train0,varianceFeature1Train0,meanFeature2Train0, varianceFeature2Train0= calculate_digit0()
    print('(No.1) Mean of feature1 for digit0', meanFeature1Train0)
    print('(No.2) Variance of feature1 for digit0', varianceFeature1Train0)
    print('(No.3) Mean of feature2 for digit0', meanFeature2Train0)
    print('(No.4) Variance of feature2 for digit0', varianceFeature2Train0)
  
        
    meanFeature1Train1,varianceFeature1Train1,meanFeature2Train1, varianceFeature2Train1= calculate_digit1()
    print('(No.5) Mean of feature1 for digit1', meanFeature1Train1)
    print('(No.6) Variance of feature1 for digit1', varianceFeature1Train1)
    print('(No.7) Mean of feature2 for digit1', meanFeature2Train1)
    print('(No.8) Variance of feature2 for digit1', varianceFeature2Train1)
    
    #-----Task 3-----------
    
    #for task 3 normalization 
    

    norm1 = stats.norm(meanFeature1Train0, varianceFeature1Train0 ** 0.5)
    norm2 = stats.norm(meanFeature2Train0, varianceFeature2Train0 ** 0.5)
    norm3 = stats.norm(meanFeature1Train1, varianceFeature1Train1 ** 0.5)
    norm4 = stats.norm(meanFeature2Train1, varianceFeature2Train1 ** 0.5)
    
    
    final_result= calculateProbabilityMatrix(test0, norm1,norm2, norm3,norm4,testData1feature1,testData2feature1)
    final_norm1= calculateProbabilityMatrix(test1, norm1,norm2, norm3,norm4,testData1feature2,testData2feature2)

    # Task4 : Calculate the accuracy
    predicted0 = np.zeros(len(test0))
    predicted1 = np.ones(len(test1))

   
    print("accuracy test 0 ", accuracy_score( final_result, predicted0))
    print("accuracy test 1",accuracy_score( final_norm1, predicted1))
   
    pass
if __name__ == '__main__':
    main()