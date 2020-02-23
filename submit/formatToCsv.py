import pandas as pd

def readMost10():
	with open('BEST&MOST10/dev-most10.arff') as fr:
		k = 0
		data = []
		w = []
		for line in fr.readlines():
			if k < 15:
				line = line.strip().split()
				if line[0] == '@ATTRIBUTE':
					w.append(line[1])
				k += 1
				continue
			line = line.strip().split(',')
			data.append(line)
		q = [[] for x in range(13)]
		for i in range(13):
			q[i] = [x[i] for x in data]
		df = pd.DataFrame({'tweetId': q[0]})
		k = 0
		for (x, y) in zip(w, q):
			if k == 0:
				k += 1
				continue
			df[w[k]] = q[k]
			k += 1
		# print (df)
		df.to_csv('devMost10.csv', index=False)

def readBest10():
	with open('BEST&MOST10/train-best10.arff') as fr:
		k = 0
		data = []
		w = []
		for line in fr.readlines():
			if k < 29:
				line = line.strip().split()
				if line[0] == '@ATTRIBUTE':
					w.append(line[1])
				k += 1
				continue
			line = line.strip().split(',')
			data.append(line)
		q = [[] for x in range(27)]
		for i in range(27):
			q[i] = [x[i] for x in data]
		df = pd.DataFrame({'tweetId': q[0]})
		k = 0
		for (x, y) in zip(w, q):
			if k == 0:
				k += 1
				continue
			df[w[k]] = q[k]
			k += 1
		df.to_csv('trainBest10.csv', index=False)		


def readBest20():
	with open('BEST&MOST20/test-best20.arff') as fr:
		k = 0
		data = []
		w = []
		for line in fr.readlines():
			if k < 52:
				line = line.strip().split()
				if line[0] == '@ATTRIBUTE':
					w.append(line[1])
				k += 1
				continue
			line = line.strip().split(',')
			data.append(line)
		q = [[] for x in range(50)]
		for i in range(50):
			q[i] = [x[i] for x in data]
		df = pd.DataFrame({'tweetId': q[0]})
		k = 0
		for (x, y) in zip(w, q):
			if k == 0:
				k += 1
				continue
			df[w[k]] = q[k]
			k += 1
		df.to_csv('testBest20.csv', index=False)	

def readBest50():
	with open('BEST&MOST50/test-best50.arff') as fr:
		k = 0
		data = []
		w = []
		for line in fr.readlines():
			if k < 127:
				line = line.strip().split()
				if line[0] == '@ATTRIBUTE':
					w.append(line[1])
				k += 1
				continue
			line = line.strip().split(',')
			data.append(line)
		q = [[] for x in range(125)]
		for i in range(125):
			q[i] = [x[i] for x in data]
		df = pd.DataFrame({'tweetId': q[0]})
		k = 0
		for (x, y) in zip(w, q):
			if k == 0:
				k += 1
				continue
			df[w[k]] = q[k]
			k += 1
		df.to_csv('testBest50.csv', index=False)	


def readBest200():
	with open('BEST&MOST200/test-best200.arff') as fr:
		k = 0
		data = []
		w = []
		for line in fr.readlines():
			if k < 459:
				line = line.strip().split()
				if line[0] == '@ATTRIBUTE':
					w.append(line[1])
				k += 1
				continue
			line = line.strip().split(',')
			data.append(line)
		q = [[] for x in range(457)]
		for i in range(457):
			q[i] = [x[i] for x in data]
		df = pd.DataFrame({'tweetId': q[0]})
		k = 0
		for (x, y) in zip(w, q):
			if k == 0:
				k += 1
				continue
			df[w[k]] = q[k]
			k += 1
		df.to_csv('testBest200.csv', index=False)


def readMost200():
	with open('BEST&MOST200/dev-most200.arff') as fr:
		k = 0
		data = []
		w = []
		for line in fr.readlines():
			if k < 205:
				line = line.strip().split()
				if line[0] == '@ATTRIBUTE':
					w.append(line[1])
				k += 1
				continue
			line = line.strip().split(',')
			data.append(line)
		q = [[] for x in range(203)]
		for i in range(203):
			q[i] = [x[i] for x in data]
		df = pd.DataFrame({'tweetId': q[0]})
		k = 0
		for (x, y) in zip(w, q):
			if k == 0:
				k += 1
				continue
			df[w[k]] = q[k]
			k += 1
		# print (df)
		df.to_csv('devMost200.csv', index=False)

def readMost50():
	with open('BEST&MOST50/train-most50.arff') as fr:
		k = 0
		data = []
		w = []
		for line in fr.readlines():
			if k < 55:
				line = line.strip().split()
				if line[0] == '@ATTRIBUTE':
					w.append(line[1])
				k += 1
				continue
			line = line.strip().split(',')
			data.append(line)
		q = [[] for x in range(53)]
		for i in range(53):
			q[i] = [x[i] for x in data]
		df = pd.DataFrame({'tweetId': q[0]})
		k = 0
		for (x, y) in zip(w, q):
			if k == 0:
				k += 1
				continue
			df[w[k]] = q[k]
			k += 1
		# print (df)
		df.to_csv('trainMost50.csv', index=False)

if __name__ == '__main__':
	# readMost10()
	# x = pd.read_csv('trainMost10.csv')
	# print (x)
	readBest200()










	
