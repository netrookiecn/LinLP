import nltk
import jieba
import jieba.posseg as k
text1 = jieba.cut("官网订购的产品,如何查询订单或物流")
jiang = k.cut("官网订购的产品,如何查询订单或物流")
print(jiang)