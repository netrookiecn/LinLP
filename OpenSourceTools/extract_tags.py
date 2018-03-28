import sys
sys.path.append('../')

import jieba.analyse

print("jieba抽取默认使用tfidf")
print("输入内容为content，top k表示抽取出的关键词个数： ")
content = "在会谈中，习近平代表中共中央对金正恩首次访问中国表示热烈欢迎。习近平表示，你在中共十九大后电贺我再次当选中共中央总书记、就任党中央军委主席，前些天又第一时间电贺我再次当选国家主席、国家中央军委主席，我对此表示感谢。此次来华访问，时机特殊、意义重大，充分体现了委员长同志和朝党中央对中朝两党两国关系的高度重视，我们对此高度评价。"
topK=3
tags = jieba.analyse.extract_tags(content, topK=topK)
print(",".join(tags))



print("如果需要计算某个单词的权重，则需要添加参数：withWeight=True")
tags = jieba.analyse.extract_tags(content, topK=topK, withWeight=True)

for tag in tags:
    print("tag: %s\t\t weight: %f" % (tag[0],tag[1]))


print("使用Textrank的计算结果为：")
for x, w in jieba.analyse.textrank(content, topK=topK,withWeight=True):
    print('%s %s' % (x, w))
