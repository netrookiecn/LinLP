import nltk
import jieba
import jieba.analyse
import jieba.posseg as ps

jieba.analyse.set_stop_words("./resource/stopwords.dic")
seg = jieba.cut("官网订购的产品,如何查询订单或物流")
words = ps.cut("官网订购的产品,如何查询订单或物流")
for word, flag in words:
    print('%s  %s'  (word, flag))
print("\n".join(seg))