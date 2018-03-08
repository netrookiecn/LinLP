from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.svm import SVC
from sentiment import JiebaWithStopwords

def build_sentence_vector(text):
    size=200
    imdb_w2v = Word2Vec.load('w2v_model.pkl')
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v.wv.__getitem__(word).reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

neg=pd.read_excel('data/neg.xls',header=None,index=None)
pos=pd.read_excel('data/pos.xls',header=None,index=None)
#分词
#cw = lambda x: list(jieba.cut(x))
cw = lambda x: list(JiebaWithStopwords.parseWithStopwords(x))
neg['words'] = neg[0].apply(cw)
pos['words'] = pos[0].apply(cw)
y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))
x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos['words'],neg['words'])), y,test_size=0.2)

print('begin word2vec')
n_dim = 200
imdb_w2v = Word2Vec(x_train,size=n_dim,window=5, min_count=5)
imdb_w2v.save('w2v_model.pkl')
print('finished word2vec')


clf = SVC(kernel='rbf', verbose=True)
train_vecs = np.concatenate([build_sentence_vector(z) for z in x_train])
print('begin')
clf.fit(train_vecs, y_train)
joblib.dump(clf, 'svmModel.pkl')
print('predict')
test_vecs = np.concatenate([build_sentence_vector(rr) for rr in x_test])
print(clf.score(test_vecs, y_test))
