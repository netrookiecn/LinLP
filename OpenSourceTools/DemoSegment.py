import jieba
import jieba.posseg as psg

#分词  无词性标注 无停止词
text1 = jieba.cut("官网订购的产品,如何查询订单或物流")
print(text1)
print(" ".join(text1))

#分词+词性标注
jiang = psg.cut("官网订购的产品,如何查询订单或物流")
for word, tag in jiang:
    print('%s %s' % (word, tag))

#分词 + 停止词
