import numpy as np
import pandas as pd
import re
from numpy import log, min

f = open('sentences.txt', 'r',encoding='utf-8')  # 读取文章
s = f.read()  # 读取为一个字符串

# 定义要去掉的标点字
drop_dict = [u'，', u'\n', u'。', u'、', u'：', u'(', u')', u'[', u']', u'.', u',', u' ', u'\u3000', u'”', u'“', u'？', u'?',
             u'！', u'‘', u'’', u'…']
for i in drop_dict:  # 去掉标点字
    s = s.replace(i, '')

# 为了方便调用，自定义了一个正则表达式的词典
myre = {2: '(..)', 3: '(...)', 4: '(....)', 5: '(.....)', 6: '(......)', 7: '(.......)'}

min_count = 10  # 录取词语最小出现次数
min_support = 30  # 录取词语最低支持度，1代表着随机组合
min_s = 3  # 录取词语最低信息熵，越大说明越有可能独立成词
max_sep = 4  # 候选词语的最大字数
t = []  # 保存结果用。

t.append(pd.Series(list(s)).value_counts())  # 逐字统计
tsum = t[0].sum()  # 统计总字数
rt = []  # 保存结果用

for m in range(2, max_sep + 1):
    print(u'正在生成%s字词...' % m)
    t.append([])
    for i in range(m):  # 生成所有可能的m字词
        t[m - 1] = t[m - 1] + re.findall(myre[m], s[i:])

    t[m - 1] = pd.Series(t[m - 1]).value_counts()  # 逐词统计
    t[m - 1] = t[m - 1][t[m - 1] > min_count]  # 最小次数筛选
    tt = t[m - 1][:]
    for k in range(m - 1):
        qq = np.array(list(map(lambda ms: tsum * t[m - 1][ms] / t[m - 2 - k][ms[:m - 1 - k]] / t[k][ms[m - 1 - k:]],
                               tt.index))) > min_support  # 最小支持度筛选。
        tt = tt[qq]
    rt.append(tt.index)


def cal_S(sl):  # 信息熵计算函数
    return -((sl / sl.sum()).apply(log) * sl / sl.sum()).sum()


for i in range(2, max_sep + 1):
    print(u'正在进行%s字词的最大熵筛选(%s)...' % (i, len(rt[i - 2])))
    pp = []  # 保存所有的左右邻结果
    for j in range(i):
        pp = pp + re.findall('(.)%s(.)' % myre[i], s[j:])
    pp = pd.DataFrame(pp).set_index(1).sort_index()  # 先排序，这个很重要，可以加快检索速度
    index = np.sort(np.intersect1d(rt[i - 2], pp.index))  # 作交集
    # 下面两句分别是左邻和右邻信息熵筛选
    index = index[np.array(list(map(lambda s: cal_S(pd.Series(pp[0][s]).value_counts()), index))) > min_s]
    rt[i - 2] = index[np.array(list(map(lambda s: cal_S(pd.Series(pp[2][s]).value_counts()), index))) > min_s]

# 下面都是输出前处理
for i in range(len(rt)):
    t[i + 1] = t[i + 1][rt[i]]
    sorted(t[i+1])

# 保存结果并输出
pd.DataFrame(pd.concat(t[1:])).to_csv('result.txt', header=False)