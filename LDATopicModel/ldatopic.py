#coding=utf-8
import numpy as np
import pandas as pd
import re

#通过正则表达式对文本进行清理
def cleantext(text):
    text = text.replace("\n"," ")
    text = re.sub(r"-"," ",text)
    text = re.sub(r"\d+/\d+/\d+","",text)
    text = re.sub(r"[0-2]?[0-9]:[0-6][0-9]","",text)
    text = re.sub(r"[\w]+@[\.\w]+","",text)
    text = re.sub(r"/[a-zA-Z]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i",",",text)
    pure_text=''
    for letter in text:
        if letter.isalpha() or letter==' ':
            pure_text += letter
    text = ' '.join(word for word in pure_text.split() if len(word)>1)
    return text

#读文件至内存
df = pd.read_csv('HillaryEmails.csv')

#预处理 将空值元素删除 因为只针对文本 这里只选择id 和 邮件主题部分
df=df[['Id','ExtractedBodyText']].dropna()

docs = df['ExtractedBodyText']
docs = docs.apply(lambda s:cleantext(s))

doclist = docs.values

print(u"已处理文本为List<String>格式..\n")

#使用gensim进行lda构建
from gensim import corpora,models,similarities
import gensim

#添加英文的停止词
stoplist = ['very', 'ourselves', 'am', 'doesn', 'through', 'me', 'against', 'up', 'just', 'her', 'ours',
            'couldn', 'because', 'is', 'isn', 'it', 'only', 'in', 'such', 'too', 'mustn', 'under', 'their',
            'if', 'to', 'my', 'himself', 'after', 'why', 'while', 'can', 'each', 'itself', 'his', 'all', 'once',
            'herself', 'more', 'our', 'they', 'hasn', 'on', 'ma', 'them', 'its', 'where', 'did', 'll', 'you',
            'didn', 'nor', 'as', 'now', 'before', 'those', 'yours', 'from', 'who', 'was', 'm', 'been', 'will',
            'into', 'same', 'how', 'some', 'of', 'out', 'with', 's', 'being', 't', 'mightn', 'she', 'again', 'be',
            'by', 'shan', 'have', 'yourselves', 'needn', 'and', 'are', 'o', 'these', 'further', 'most', 'yourself',
            'having', 'aren', 'here', 'he', 'were', 'but', 'this', 'myself', 'own', 'we', 'so', 'i', 'does', 'both',
            'when', 'between', 'd', 'had', 'the', 'y', 'has', 'down', 'off', 'than', 'haven', 'whom', 'wouldn',
            'should', 've', 'over', 'themselves', 'few', 'then', 'hadn', 'what', 'until', 'won', 'no', 'about',
            'any', 'that', 'for', 'shouldn', 'don', 'do', 'there', 'doing', 'an', 'or', 'ain', 'hers', 'wasn',
            'weren', 'above', 'a', 'at', 'your', 'theirs', 'below', 'other', 'not', 're', 'him', 'during', 'which']


textlists = [[word for word in d.lower().split() if word not in stoplist] for d in doclist]
print(u"已处理文本为List<List<String>>格式..\n")
print(textlists[0])

#对所有词建立词典  很强大的功能 建立词袋模型
dictionary = corpora.Dictionary(textlists)
corpus = [dictionary.doc2bow(text) for text in textlists]

print(u"已处理文本为词袋模型 每个文档将变为id和频数 \n")
print(corpus[12])

#建立lda模型 参数为语料 词典 主题数
lda = gensim.models.LdaModel(corpus=corpus,id2word=dictionary,num_topics=20)

lda.save("lda.model")
print(u"主题模型建立完毕..\n")

print(u"打印单个文档主题..\n")
print(lda.print_topic(1,topn=4))

print(u"打印全部文档主题..\n")
print(lda.print_topics(num_topics=20,num_words=5))
