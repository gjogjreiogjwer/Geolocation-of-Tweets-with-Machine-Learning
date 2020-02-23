import re
from kashgari.tasks.classification import CNNLSTMModel
import kashgari
import pandas as pd



def stop_word(x):
	a = ['tweetId', 'user-id', 'about', 'all', 'am', 'as', 
		'at', 'and', 'are', 'a', 'b', 'been', 'by', 'can', 
		'd', 'do', 'for', 'get', 'going', 'gone', 'has', 
		'have', 'i', 'if', 'in', 'is', 'it', 'its', 'know', 
		'my', 'n', 'of', 'on', 'p', 'should', 'that', 'the', 
		'they', 'think', 'this', 'to', 'u', 'well', 'were', 
		'what', 'when', 'will', 'with', 'would', 'you']
	for i in a:
		x = re.sub(i, "", x)
	return x

def load(file):
	with open('tweets/%s_tweets.txt' % file, encoding='latin-1') as fr:
		content = []
		label = []
		pattern = re.compile('"(.*)"')
		for line in fr.readlines():
			valid = []
			line = line.strip()
			label.append(line.split(',')[-1])
			x = pattern.findall(line)[0]
			x = x.lower()
			x = re.sub(r"[^A-Za-z0-9^,!.\/'+-=@\_]", " ", x)
			x = re.sub(r"what's", "what is ", x)
			x = re.sub(r"\'s", " ", x)
			x = re.sub(r"\'ve", " have ", x)
			x = re.sub(r"n't", " not ", x)
			x = re.sub(r"i'm", "i am ", x)
			x = re.sub(r"\'re", " are ", x)
			x = re.sub(r"\'d", " would ", x)
			x = re.sub(r"\'ll", " will ", x)
			x = re.sub(r",", "", x)
			x = re.sub(r"\.", "", x)
			x = re.sub(r"!", "", x)
			x = re.sub(r"\?", "", x)
			x = re.sub(r"\/", "", x)
			x = re.sub(r"\^", "", x)
			x = re.sub(r"\+", "", x)
			x = re.sub(r"\-", "", x)
			x = re.sub(r"\=", "", x)
			x = re.sub(r"'", "", x)
			x = stop_word(x)
			for word in x.split():
				if not word.startswith('@'):
					valid.append(word)
			content.append(valid)
		# print (content)
		# print (label)
	return content, label

trainX, trainY = load('train')
devX, devY = load('dev')
testX, testY = load('test')
# print (trainX[:20])

def trainModel():
	model = CNNLSTMModel()
	model.fit(trainX, trainY, batch_size=16, epochs=5)
	predictions = model.predict(devX)
	count = 0
	for (i,j) in zip(predictions, devY):
		if i == j:
			count += 1
	print (count/len(predictions))
	return predictions
	


def getResult():
	testDf = pd.read_csv('data/testBest20.csv')
	tweetId = testDf.tweetId
	predictions = trainModel()
	result = pd.DataFrame({'tweet-id': tweetId, 'class': predictions})
	result.to_csv('result1.csv', index=False)


# 0.6472

# print (trainX[:20])

# getResult()


trainModel()


















