import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import model_selection
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import accuracy_score
from math import log
import operator
import graphviz
from sklearn.externals.six import StringIO
import pydot


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)

def setLabel(x):
	if x == 'NewYork':
		return 0
	if x == 'California':
		return 1
	if x == 'Georgia':
		return 2

trainDf = pd.read_csv('data/trainBest200.csv')
trainDf.drop(['tweetId', 'user-id', 'about', 'ab', 'all', 'am', 'as', 'at', 'and', 'are', 'a', 'as', 'as', 'b', 'be', 'been', 'by', 'can', 'd', 'da', 'do', 'fa', 'for', 'get', 'ga', 'going', 'gone', 'has', 'have', 'i', 'if', 'in', 'is', 'it', 'its', 'know', 'just', 'la', 'lt', 'ma', 'make', 'more', 'my', 'n', 'na', 'of', 'on', 'p', 'should', 'that', 'the', 'they', 'think', 'this', 'to', 'u', 'well', 'were', 'what', 'when', 'will', 'with', 'would', 'you'], axis=1, inplace=True)
# trainDf = trainDf.filter(regex='a|^i$|im|the|to|u|class')
# trainDf['class'] = trainDf['class'].apply(setLabel)
# print (trainDf.head())
trainMat = trainDf.values
trainX = trainMat[:, :-1]
trainY = trainMat[:, -1]

testDf = pd.read_csv('data/devBest200.csv')
testDf.drop(['tweetId', 'user-id', 'about', 'ab', 'all', 'am', 'as', 'at', 'and', 'are', 'a', 'as', 'as', 'b', 'be', 'been', 'by', 'can', 'd', 'da', 'do', 'fa', 'for', 'get', 'ga', 'going', 'gone', 'has', 'have', 'i', 'if', 'in', 'is', 'it', 'its', 'know', 'just', 'la', 'lt', 'ma', 'make', 'more', 'my', 'n', 'na', 'of', 'on', 'p', 'should', 'that', 'the', 'they', 'think', 'this', 'to', 'u', 'well', 'were', 'what', 'when', 'will', 'with', 'would', 'you'], axis=1, inplace=True)
# # testDf = testDf.filter(regex='a|^i$|im|the|to|u|class')
# testDf.label = testDf['class'].apply(setLabel)
testMat = testDf.values
testX = testMat[:, :-1]
testY = testMat[:, -1]


a = list(trainMat)
a = [list(x) for x in a]
# print (a)
b = list(trainDf.columns[:-1])



def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for feat in dataSet:
        current = feat[-1]
        labelCounts[current] = labelCounts.get(current,0) + 1
    shannonEnt = 0
    for key in labelCounts:
        prob = labelCounts[key]/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for feat in dataSet:
        if feat[axis] == value:
            #去掉feat[axis]
            reducedFeat = feat[:axis]
            reducedFeat.extend(feat[axis+1:])
            retDataSet.append(reducedFeat)
    return retDataSet



def chooseBestFeatureToSplit(dataSet, label):
    numOfFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0
    bestFeature = -1
    x = {}
    for i in range(numOfFeatures):
        #选择一项特征
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0
        # 计算每种划分方式的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/len(dataSet)
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        x[i] = infoGain
    t = sorted(x.items(), key=operator.itemgetter(1), reverse=True)
    tt = [x[0] for x in t]
    return [label[x] for x in tt]





print (chooseBestFeatureToSplit(a, b))






x = ['haha', 'inhighschool', 'you', 'lmaoo', 'lml', 'hahaha', 'da', 'hella', 'lmaooo', 'rt', 'the', 
'wat', 'and', 'ii', 'are', 'atl', 'ya', 'that', 'smh', 'u', 'dead', 'dat', 'atlanta', 'iight', 
'will', 'dis', 'just', 'willies', 'deadass', 'ga', 'finna', 'la', 'of', 'a', 'know', 'wassup', 
'dha', 'it', 'skool', 'loll', 'have', 'oovoo', 'ma', 'lmfaoo', 'what', 'bomb', 'gw', 'lols', 
'lmfaooo', 'childplease', 'lmaoooo', 'nah', 'nd', 'flirty', 'to', 'coo', 'naw', 'dats', 'odee', 
'od', 'with', 'san', 'shyt', 'this', 'ahaha', 'mad', 'madd', 'is', 'bbm', 'thatisall', 'hahahaha', 
'lowkey', 'famu', 'fab', 'for', 'hahah', 'd', 'aha', 'lolss', 'cuhz', 'neva', 'i', 'smfh', 'has', 
'mi', 'af', 'foo', 'frequency', 'ive', 'son', 'juss', 'np', 'rain', 'gsu', 'tinos', 'parody', 
'noe', 'dhat', 'spring', 'icheatedbecause', 'lmfaoooo', 'dem', 'train', 'que', 'famusextape', 
'wet', 'smhh', 'tone', 'at', 'sed', 'fasho', 'can', 'one', 'miami', 'andthenwehadsex', 'omw', 
'spelman', 'quote', 'do', 'dang', 'gud', 'dere', 'qo', 'n', 'wen', 'sb', 'thanks', 'brooklyn', 
'ova', 'gravity', 'niggas', 'well', 'knw', 'ooo', 'def', 'ltlt', 'dom', 'neighbors', 'tha', 'bout', 
'aja', 'rapdishes', 'they', 'in', 'bruh', 'bk', 'crib', 'cau', 'bus', 'break', 'hehe', 'girl', 'cuh', 
'tho', 'oo', 'ion', 'when', 'ada', 'banget', 'nya', 'udah', 'its', 'chillen', 'uu', 'lmaooooo', 'dey', 
'nuttin', 'back', 'boaw', 'pcb', 'wwwdormtainmentcom', 'hollywood', 'hahaa', 'please', 'get', 'aye', 
'nuffin', 'lt', 'think', 'ill', 'sir', 'meh', 'doin', 'retreat', 'should', 'raining', 'tacos', 'vegas', 
'chillin', 'would', 'rite', 'btwn', 'taco', 'queens', 'na', 'den', 'dormtainment', 'mangoville', 'bay', 
'ahah', 'hell', 'niqqa', 'betta', 'djb', 'kalo', 'kno', 'jus', 'ave', 'ish', 'california', 'ahahaha', 
'cali', 'tweet', 'qot', 'hahahahahaha', 'be', 'harlem', 'folks', 'about', 'follow', 'going', 'my', 
'smacked', 'all', 'aint', 'if', 'damn', 'guys', 'alice', 'ight', 'fone', 'were', 'buggin', 'flow', 
'bc', 'car', 'jk', 'dnt', 'party', 'dope', 'hahahahaha', 'jets', 'ny', 'more', 'lmmfao', 'koo', 'by', 
'mah', 'jamaica', 'bitches', 'lolsz', 'tew', 'bt', 'lmfaooooo', 'thatswhyileftyou', 'boii', 'papi', 'as',
 'freaknik', 'rpl', 'myy', 'dum', 'mau', 'oakland', 'amazing', 'bisa', 'gloria', 'pengen', 'trevi', 'area', 
 'cablevision', 'tape', 'ahahah', 'ca', 'gone', 'trippin', 'howyouamanbut', 'on', 'patna', 'hun', 'man', 'am', 
 'coffee', 'auc', 'crisantasbreast', 'followdormtainment', 'glad', 'sextape', 'random', 'b', 'boutta', 'nun', 
 'great', 'sih', 'been', 'oh', 'apa', 'daygo', 'lagi', 'baow', 'jst', 'drive', 'iono', 'make', 'may', 'club', 
 'airport', 'xd', 'hahahah', 'diego', 'huh', 'lord', 'factsaboutme', 'ilovefamu', 'offdabonz', 'savannah', 
 'gas', 'headed', 'porter', 'lmbo', 'iya', 'p', 'shoot', 'bra', 'nigka', 'pra', 'replylt', 'shotout', 'siapa', 
 'students', 'rofl', 'pretty', 'bro', 'fxck', 'thang', 'bucks', 'duke', 'laker', 'hmu', 'blog', 'traffic', 
 'sucks', 'mall', 'aite', 'fa', 'florida', 'augusta', 'shitniggasdo', 'shyd', 'bahaha', 'brodie', 'gon', 
 'dreads', 'move', 'certifiedfreak', 'gak', 'ltreply', 'factaboutme', 'dam', 'freeway', 'ratchet', 'coolin', 
 'icheated', 'game', 'travis', 'slick', 'homie', 'refs', 'itu', 'sf', 'bahahaha', 'randomthoughts', 'sumbdy', 
 'dodgers', 'roscoes', 'state', 'posted', 'maam', 'video', 'ahahahaha', 'pride', 'dork', 'santa', 'beenhad', 
 'dormtainmentcom', 'flystyle', 'gamecocks', 'godisgood', 'goshooturself', 'hosolo', 'ifurcaribbean', 
 'playersprayer', 'shaiimillzsays', 'shid', 'texano', 'anaheim', 'backinmiddleschool', 'enimal', 'jadi', 'jg', 
 'juga', 'kenapa', 'lmfpo', 'ohok', 'socal', 'panama', 'natural', 'midterms', 'station', 'drunk', 'georgia', 
 'cont', 'bgt', 'basic', 'kissing', 'lego', 'name', 'flex', 'apoyo', 'dennys', 'lenox', 'function', 'resting', 
 'strip', 'commons', 'donnie', 'subliminaltweet', 'suburbs', 'campus', 'flexin', 'coon', 'chrisette', 'shi', 
 'buckhead', 'dontsaywhenhavingsex', 'fye', 'gsusextape', 'hwy', 'inpanamacity', 'killyoself', 'marta', 'myexis', 
 'reasonitwontwork', 'services', 'ssucookout', 'stonecrest', 'ab', 'becuz', 'bae', 'ifuckmymoneyupnowicantreup', 
 'leggo', 'operation', 'pageant', 'christ']

print (len(x))

# top 50
a = ['haha', 'inhighschool', 'lmaoo', 'lml', 'da', 'hella', 'rt', 
'wat', 'ii', 'atl', 'ya', 'smh', 'dead', 'dat', 'atlanta', 'iight','dis', 'willies', 
'deadass', 'ga', 'finna', 'la', 'wassup', 'dha', 'skool', 'loll', 'oovoo', 'ma', 'lmfaoo', 
'bomb', 'gw', 'lols','childplease', 'nah', 'nd', 'flirty', 'coo', 'naw', 'dats', 'odee',
'san', 'shyt', 'madd', 'bbm', 'thatisall', 'lowkey', 'famu', 'fab', 'aha', 'lolss']















































