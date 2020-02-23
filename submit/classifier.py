from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import naive_bayes
from sklearn import model_selection
import matplotlib.pyplot as plt
from useridInformation import *
from scipy import stats
from sklearn import linear_model
from sklearn import tree
import math
from featureSelection import *
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier


def setLabel(x):
	if x == 'NewYork':
		return 0
	if x == 'California':
		return 1
	if x == 'Georgia':
		return 2
# TF-IDF
def tfIdf():
	trainDf = pd.read_csv('data/trainBest50.csv')
	trainDf.drop(['tweetId', 'user-id', 'and', 'are', 'a', 'ga', 'has', 'have', 'is', 'it', 'know', 'just', 'la', 'ma', 'of', 'that', 'the', 'this', 'to', 'u',  'what', 'will', 'with'], axis=1, inplace=True)
	trainDf['class'] = trainDf['class'].apply(setLabel)
	testDf = pd.read_csv('data/devBest50.csv')
	# testDf.drop(['tweetId', 'user-id', 'about', 'ab', 'all', 'am', 'as', 'at', 'and', 'are', 'a', 'as', 'as', 'b', 'be', 'been', 'by', 'can', 'd', 'da', 'do', 'fa', 'for', 'get', 'ga', 'going', 'gone', 'has', 'have', 'i', 'if', 'in', 'is', 'it', 'its', 'know', 'just', 'la', 'lt', 'ma', 'make', 'more', 'my', 'n', 'na', 'of', 'on', 'p', 'should', 'that', 'the', 'they', 'think', 'this', 'to', 'u', 'well', 'were', 'what', 'when', 'will', 'with', 'would', 'you'], axis=1, inplace=True)
	testDf.drop(['tweetId', 'user-id', 'and', 'are', 'a', 'ga', 'has', 'have', 'is', 'it', 'know', 'just', 'la', 'ma', 'of', 'that', 'the', 'this', 'to', 'u',  'what', 'will', 'with'], axis=1, inplace=True)
	testDf['class'] = testDf['class'].apply(setLabel)

	testMat = testDf.values
	testX = testMat[:, :-1]
	frequencySum = testX.sum(axis=0)
	features = testDf.columns.tolist()
	count = 0
	x = []
	for i in frequencySum:
		if i == 0:
			x.append(count)
		count += 1
	for i in range(len(x)):
		trainDf.drop([features[x[i]]], axis=1, inplace=True)
		testDf.drop([features[x[i]]], axis=1, inplace=True)

	trainMat = trainDf.values
	trainX = trainMat[:, :-1]
	trainY = trainMat[:, -1]
	testMat = testDf.values
	testX = testMat[:, :-1]
	testY = testMat[:, -1]

	frequencySum = trainX.sum(axis=0)
	sampleNum = len(trainX)
	idf = sampleNum / frequencySum
	trainX = np.multiply(trainX, idf)

	frequencySum = testX.sum(axis=0)
	sampleNum = len(testX)
	idf = log(sampleNum / frequencySum, 2)
	testX = np.multiply(testX, idf)

	std = MinMaxScaler()
	testX = std.fit_transform(testX)
	trainX = std.fit_transform(trainX)

	bayes = naive_bayes.MultinomialNB()
	bayes.fit(trainX, trainY)
	predictions = bayes.predict(testX)
	accuracy = bayes.score(testX, testY)
	print (accuracy)

# find all twitters for a user, results recorded in useridInformation.py
def find():
	testDf = pd.read_csv('data/trainBest200.csv')
	users = {}
	x = set(testDf['user-id'].tolist())
	testMat = testDf.values
	# print (testDf.iloc[0, 1])
	for i in range(len(testMat)):
		for k in x:
			if k == testDf.iloc[i, 1]:
				if k not in users:
					users[k] = []
				users[k].append(i)
				break
	print (users)

def classify():
	trainDf = pd.read_csv('data/trainBest200.csv')
	# trainDf.drop(['tweetId', 'user-id', 'about', 'ab', 'all', 'am', 'as', 'at', 'and', 'are', 'a', 'as', 'as', 'b', 'be', 'been', 'by', 'can', 'd', 'da', 'do', 'fa', 'for', 'get', 'ga', 'going', 'gone', 'has', 'have', 'i', 'if', 'in', 'is', 'it', 'its', 'know', 'just', 'la', 'lt', 'ma', 'make', 'more', 'my', 'n', 'na', 'of', 'on', 'p', 'should', 'that', 'the', 'they', 'think', 'this', 'to', 'u', 'well', 'were', 'what', 'when', 'will', 'with', 'would', 'you'], axis=1, inplace=True)
	# trainDf.drop(['tweetId', 'user-id', 'about', 'ab', 'all', 'am', 'as', 'at', 'and', 'are', 'a', 'as', 'as', 'be', 'been', 'by', 'can', 'da', 'do', 'fa', 'for', 'get', 'ga', 'going', 'gone', 'has', 'have', 'i', 'if', 'in', 'is', 'it', 'its', 'know', 'just', 'la', 'lt', 'ma', 'make', 'more', 'my', 'na', 'of', 'on', 'should', 'that', 'the', 'they', 'think', 'this', 'to', 'well', 'were', 'what', 'when', 'will', 'with', 'would', 'you'], axis=1, inplace=True)
	
	testDf = pd.read_csv('data/devBest200.csv')
	# testDf.drop(['tweetId', 'user-id', 'about', 'ab', 'all', 'am', 'as', 'at', 'and', 'are', 'a', 'as', 'as', 'b', 'be', 'been', 'by', 'can', 'd', 'da', 'do', 'fa', 'for', 'get', 'ga', 'going', 'gone', 'has', 'have', 'i', 'if', 'in', 'is', 'it', 'its', 'know', 'just', 'la', 'lt', 'ma', 'make', 'more', 'my', 'n', 'na', 'of', 'on', 'p', 'should', 'that', 'the', 'they', 'think', 'this', 'to', 'u', 'well', 'were', 'what', 'when', 'will', 'with', 'would', 'you'], axis=1, inplace=True)
	
	trainDf = trainDf[top150 + ['class']]
	testDf = testDf[top150 + ['class']]

	trainDf['class'] = trainDf['class'].apply(setLabel)
	testDf['class'] = testDf['class'].apply(setLabel)
	trainMat = trainDf.values
	trainX = trainMat[:, :-1]
	trainY = trainMat[:, -1]
	testMat = testDf.values
	testX = testMat[:, :-1]
	testY = testMat[:, -1]
	newTestX = []
	for k in users:
		tempTestX = np.zeros((1, testX.shape[1]))
		tempTestY = 0
		for x in users[k]:
			tempTestX += testX[x]
		newTestX.append(tempTestX.tolist()[0])
	newTestX = np.array(newTestX)
	model = naive_bayes.MultinomialNB()
	# log = linear_model.LogisticRegression(solver='sag', multi_class='multinomial', class_weight='balanced', C=20)
	# model = tree.DecisionTreeClassifier()
	# mlp = MLPClassifier(activation='relu', solver='adam', alpha=0.0001)
	# knn = neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=1) 
	# model = BaggingClassifier( bayes, n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1 )
	model.fit(trainX, trainY)
	predictions = model.predict(newTestX)
	newPredictions = np.zeros((testY.shape[0], 1))
	i = 0
	for k in users:
		for x in users[k]:
			newPredictions[x] = predictions[i]
		i += 1
	count = 0
	for x,y in zip(newPredictions, testY):
		if x == y:
			count += 1
	print (count/len(testX))
	TP = 0
	FP = 0
	FN = 0
	for x,y in zip(newPredictions, testY):
		if x == y and x == 2:
			TP += 1
		if x == 2 and y != 2:
			FP += 1
		if x != 2 and y == 2:
			FN += 1
	print ('precision: ', TP/(TP+FP))
	print ('recall: ', TP/(TP+FN))
# classify()


def result():
	trainDf = pd.read_csv('data/trainBest200.csv')
	trainDf.drop(['tweetId', 'user-id', 'about', 'ab', 'all', 'am', 'as', 'at', 'and', 'are', 'a', 'as', 'as', 'b', 'be', 'been', 'by', 'can', 'd', 'da', 'do', 'fa', 'for', 'get', 'ga', 'going', 'gone', 'has', 'have', 'i', 'if', 'in', 'is', 'it', 'its', 'know', 'just', 'la', 'lt', 'ma', 'make', 'more', 'my', 'n', 'na', 'of', 'on', 'p', 'should', 'that', 'the', 'they', 'think', 'this', 'to', 'u', 'well', 'were', 'what', 'when', 'will', 'with', 'would', 'you'], axis=1, inplace=True)	
	trainDf = trainDf[top150 + ['class']]
	trainDf['class'] = trainDf['class'].apply(setLabel)
	trainMat = trainDf.values
	trainX = trainMat[:, :-1]
	trainY = trainMat[:, -1]
	testDf = pd.read_csv('data/testBest200.csv')
	tweetId = testDf.tweetId
	testDf.drop(['tweetId', 'user-id', 'about', 'ab', 'all', 'am', 'as', 'at', 'and', 'are', 'a', 'as', 'as', 'b', 'be', 'been', 'by', 'can', 'd', 'da', 'do', 'fa', 'for', 'get', 'ga', 'going', 'gone', 'has', 'have', 'i', 'if', 'in', 'is', 'it', 'its', 'know', 'just', 'la', 'lt', 'ma', 'make', 'more', 'my', 'n', 'na', 'of', 'on', 'p', 'should', 'that', 'the', 'they', 'think', 'this', 'to', 'u', 'well', 'were', 'what', 'when', 'will', 'with', 'would', 'you'], axis=1, inplace=True)
	testDf = testDf[top150 + ['class']]
	testDf['class'] = testDf['class'].apply(setLabel)
	testMat = testDf.values
	testX = testMat[:, :-1]
	testY = testMat[:, -1]
	newTestX = []
	for k in users1:
		tempTestX = np.zeros((1, testX.shape[1]))
		tempTestY = 0
		for x in users1[k]:
			tempTestX = np.add(tempTestX, testX[x])
		newTestX.append(tempTestX.tolist()[0])
	newTestX = np.array(newTestX)

	bayes = naive_bayes.MultinomialNB()
	bayes.fit(trainX, trainY)
	predictions = bayes.predict(newTestX)
	newPredictions = np.zeros((testY.shape[0], 1))
	i = 0
	for k in users1:
		for x in users1[k]:
			newPredictions[x] = predictions[i]
		i += 1
	finalPredicitons = []
	m = 0
	for prediction in newPredictions:
		if prediction[0] == 0:
			finalPredicitons.append('NewYork')
		if prediction[0] == 1:
			finalPredicitons.append('California')
		if prediction[0] == 2:
			finalPredicitons.append('Georgia')
		m += 1

	result = pd.DataFrame({'tweet-id': tweetId, 'class': finalPredicitons})
	result.to_csv('result10.csv', index=False)
# result()
















