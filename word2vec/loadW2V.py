#coding=utf-8
from gensim.models.word2vec import Word2Vec
import numpy as np

word="test"
#size为向量维度
size = 200
w2v = Word2Vec.load("dbw2v.model")
vec = w2v.wv.__getitem__(word).reshape((1,size))