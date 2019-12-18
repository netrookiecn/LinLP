# 使用Python进行自然语言处理实践

实验环境：
python --version 3.5.2  
tensorflow  
keras  


本项目旨在归纳总结自然语言处理工程师所需的理论与实践内容，每个模块尽量包含理论（经典算法和最新paper）和实践部分。


## 一、基础理论部分

1、基础数学理论：

线性代数、概率论

2、机器学习：

逻辑回归、贝叶斯、k近邻、决策树、最大熵、支持向量机、EM、GBDT、XGBoost

3、深度学习：

3.1 CNN  
3.2 RNN   
3.3 归一化  
3.4 激活函数  
3.5 dropout  
3.6  反向传播

4、概率图模型：

隐马尔科夫、条件随机场

5、强化学习：

基于策略梯度、基于价值

### 二、常见自然语言处理任务

1、文本向量化：

1.1 tfidf

1.2 word2vec  
（文件夹）word2vec:word2vec使用（done）

1.3 glove  
1.4 elmo  
1.5 bert  
1.6 xlnet  

2、文本分类问题

2.1（文件夹）sentiment：情感分析实践-目前使用word2vec和svm实现（done）  
2.2 BiLSTM  
2.3 HAN  
 
3、自然语言推理问题  
相似文本匹配、问答匹配）



4、语言生成问题

闲聊、摘要

5、任务型对话
对话状态跟踪、对话策略生成、用户模拟器


6、知识图谱
基于neo4j构建问答系统
transe

7、信息抽取
命名实体识别


8、自然语言基础功能：
8.1 分词

8.2 词性标注
（文件夹）HMMPostag:hmm词性标注（done）

8.3 句法分析

8.4 新词发现
（文件夹）NewWordFinder:新词发现功能（done）

8.5 指代消解


9、文本聚类

（文件夹）LDATopicModel:lda主题模型（done）





备注：
（文件夹）resource：部分项目公共资源目录



