import jieba
def stopwordsList(filepath):
    stopwords = [line.strip() for line in open(filepath,'r',encoding='utf-8').readlines()]
    return stopwords
def parseWithStopwords(str):
    stopwords = stopwordsList('stopwords.txt')
    k = jieba.cut(str,cut_all=False)
    output = []
    for word in k:
        if word not in stopwords:
            if word != '\t':
                output.append(word)
    return output
def dict2list(dic:dict):
    ''' 将字典转化为列表 '''
    keys = dic.keys()
    vals = dic.values()
    lst = [(key, val) for key, val in zip(keys, vals)]
    return lst

