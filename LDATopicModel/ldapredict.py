#使用gensim进行lda构建
from gensim import corpora,models,similarities
import gensim

lda = gensim.models.LdaModel.load("lda.model")
print(lda.print_topic(1,4)+"\n")

dictionary = corpora.Dictionary.load("lda.model.id2word")

print(dictionary.doc2bow(['roger', 'see']))
print("\n")
#对于新句子 必须先处理为词袋表示  lda.get_document_topics(bow)  lda.get_term_topics(word_id)

print("\n根据词袋模型表示的文档 计算其属于所有主题的可能性 输出概率值: ")
print(lda.get_document_topics(dictionary.doc2bow(['roger', 'see'])))
print("\n根据某个单词的id 预测其属于哪个主题的概率 可以设定最小概率 避免没有结果 返回list:")
print(lda.get_term_topics(3208,minimum_probability=0.0000000001))
