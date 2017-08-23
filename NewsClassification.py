from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB

#加载20个分类的文档
newsgroup_train = fetch_20newsgroups()
#输出类别
pprint(list(newsgroup_train.target_names))
#使用fit_transform进行特征提取
vectorizer = HashingVectorizer(stop_words='english',non_negative =True, n_features=10000)
#训练及测试数据
fea_train = vectorizer.fit_transform(newsgroup_train.data)
#fea_test  = vectorizer.fit_transform(newsgroup_test.data)
print(repr(fea_train))

#使用分类器进行分类
from sklearn import metrics
clf = MultinomialNB(alpha=0.01)
clf.fit(fea_train,newsgroup_train.target_names)