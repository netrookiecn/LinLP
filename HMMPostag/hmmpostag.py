#coding=utf-8
#使用nltk自带数据
import nltk
from nltk.corpus import brown

#词库预处理
brown_tags_words = []

#希望处理成以下所示的结构
#('START', 'START'), ('AT', 'A'), ('JJ', 'strange'), ('NN', 'relationship'), ('IN', 'between'), ('NP', 'Joan'), ('NP', 'Fulbright'), ('CC', 'and'), ('PP', 'himself'), ('.', '.'), ('END', 'END'),

#brown.tagged_sents()为带词性的句子

for sent in brown.tagged_sents():
    brown_tags_words.append(("START","START"))
    brown_tags_words.extend([(tag[:2],word) for (word,tag) in sent])
    brown_tags_words.append(("END","END"))


print("处理完之后的数据格式如下: \n")
print(brown_tags_words)

print("统计单词与词性之间的共现的概率（该词和词性同时出现的个数 / 总的词和词性数）：")

cfd_tagwords = nltk.ConditionalFreqDist(brown_tags_words)
cpd_tagwords = nltk.ConditionalProbDist(cfd_tagwords,nltk.MLEProbDist)

print("VB 和 shut同时出现的概率： ")
print(cpd_tagwords["VB"].prob("shut"))

print("除了计算词和词性的共现概率之外，还需要计算二元模型，也就是相邻两个词性的共现概率")
#取出所有词性标签
brown_tags = [tag for (tag,word) in brown_tags_words]
cfd_tags = nltk.ConditionalFreqDist(nltk.bigrams(brown_tags))
cpd_tags=nltk.ConditionalProbDist(cfd_tags,nltk.MLEProbDist)

print("VB 和 NN 词性标签同时出现的概率： ")
print(cpd_tags["VB"].prob("NN"))

print("词性标注的任务就是：给一个句子，计算出概率最大的词性序列")

print("统计全部可能的词性标签")
alltags=set(brown_tags)

print("viterbi and backpointer（记录词性标签的前一个词性标签）")
viterbi = []
backpointer = []

first_viterbi = {}
first_pointer = {}
print("以 i want to race为例")
sentence = ["I","want","to","race"]
sentlen = len(sentence)

print("开始概率计算：")
for tag in alltags:
    if tag=="START":continue
    first_viterbi[tag] = cpd_tags["START"].prob(tag) * cpd_tagwords[tag].prob(sentence[0])
    first_pointer[tag] = "START"

print("计算第一个单词在所有标签情况下的概率，并存储到viterbi 和 pointer")
viterbi.append(first_viterbi)
backpointer.append(first_pointer)
currbest = max(first_viterbi.keys(), key = lambda tag: first_viterbi[ tag ])
print( "Word", "'" + sentence[0] + "'", "current best two-tag sequence:", first_pointer[ currbest], currbest)
print("开始计算全部单词的词性标签: ")
for wordindex in range(1,sentlen):
    this_viterbi={}
    this_pointer={}
    prev_viterbi=viterbi[-1]

    for tag in alltags:
        if tag=="START":continue
        best_previous = max(prev_viterbi.keys(),
                            key=lambda prevtag: \
                            prev_viterbi[prevtag] * cpd_tags[prevtag].prob(tag) * cpd_tagwords[tag].prob(sentence[wordindex]))
        this_viterbi[tag] = prev_viterbi[best_previous]*\
            cpd_tags[best_previous].prob(tag) * cpd_tagwords[tag].prob(sentence[wordindex])

        this_pointer[tag]=best_previous

    # 每次找完Y 我们把目前最好的 存一下
    currbest = max(this_viterbi.keys(), key=lambda tag: this_viterbi[tag])
    print("Word", "'" + sentence[wordindex] + "'", "current best two-tag sequence:", this_pointer[currbest],
              currbest)

    # 完结
    # 全部存下来
    viterbi.append(this_viterbi)
    backpointer.append(this_pointer)


# 找所有以END结尾的tag sequence
prev_viterbi = viterbi[-1]
best_previous = max(prev_viterbi.keys(),
                    key = lambda prevtag: prev_viterbi[ prevtag ] * cpd_tags[prevtag].prob("END"))

prob_tagsequence = prev_viterbi[ best_previous ] * cpd_tags[ best_previous].prob("END")

# 我们这会儿是倒着存的。。。。因为。。好的在后面
best_tagsequence = [ "END", best_previous ]
# 同理 这里也有倒过来
backpointer.reverse()

current_best_tag = best_previous
for bp in backpointer:
    best_tagsequence.append(bp[current_best_tag])
    current_best_tag = bp[current_best_tag]


best_tagsequence.reverse()
print( "The sentence was:", end = " ")
for w in sentence: print( w, end = " ")
print("\n")
print( "The best tag sequence is:", end = " ")
for t in best_tagsequence: print (t, end = " ")
print("\n")
print( "The probability of the best tag sequence is:", prob_tagsequence)


