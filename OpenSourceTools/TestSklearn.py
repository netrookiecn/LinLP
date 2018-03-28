from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
import  urllib.request
import nltk
import time
import numpy as np
#url ="http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
#下载文件并用“，”分割为矩阵用于运算
#raw_data = urllib.request.urlopen(url)
dataset = np.loadtxt("../resource/pima-indians-diabetes.data.txt", delimiter=",")

# separate the data from the target attributes

X = dataset[:, 0:7]

#X=preprocessing.MinMaxScaler().fit_transform(x)

print(X)

y = dataset[:, 8]
print(y)
print("\n调用scikit的朴素贝叶斯算法包GaussianNB ")

model = GaussianNB()

start_time = time.time()

model.fit(X, y)

print('training took %fs!' % (time.time() - start_time))

print(model)

expected = y

predicted = model.predict(X)

print(metrics.classification_report(expected, predicted))

print("分割")

print(metrics.confusion_matrix(expected, predicted))
