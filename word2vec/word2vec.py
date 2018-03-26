#coding=utf-8

"""
Produce word vectors with deep learning via word2vec's "skip-gram and CBOW models", using either
hierarchical softmax or negative sampling [1]_ [2]_.

NOTE: There are more ways to get word vectors in Gensim than just Word2Vec.
See wrappers for FastText, VarEmbed and WordRank.

The training algorithms were originally ported from the C package https://code.google.com/p/word2vec/
and extended with additional functionality.

For a blog tutorial on gensim word2vec, with an interactive web app trained on GoogleNews,
visit http://radimrehurek.com/2014/02/word2vec-tutorial/

**Make sure you have a C compiler before installing gensim, to use optimized (compiled) word2vec training**
(70x speedup compared to plain NumPy implementation [3]_).

Initialize a model with e.g.::

    >>> model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)

Persist a model to disk with::

    >>> model.save(fname)
    >>> model = Word2Vec.load(fname)  # you can continue training with the loaded model!

The word vectors are stored in a KeyedVectors instance in model.wv.
This separates the read-only word vector lookup operations in KeyedVectors from the training code in Word2Vec::

  >>> model.wv['computer']  # numpy vector of a word
  array([-0.00449447, -0.00310097,  0.02421786, ...], dtype=float32)

The word vectors can also be instantiated from an existing file on disk in the word2vec C format
as a KeyedVectors instance::

    NOTE: It is impossible to continue training the vectors loaded from the C format because hidden weights,
    vocabulary frequency and the binary tree is missing::

        >>> from gensim.models.keyedvectors import KeyedVectors
        >>> word_vectors = KeyedVectors.load_word2vec_format('/tmp/vectors.txt', binary=False)  # C text format
        >>> word_vectors = KeyedVectors.load_word2vec_format('/tmp/vectors.bin', binary=True)  # C binary format


You can perform various NLP word tasks with the model. Some of them
are already built-in::

  >>> model.wv.most_similar(positive=['woman', 'king'], negative=['man'])
  [('queen', 0.50882536), ...]

  >>> model.wv.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])
  [('queen', 0.71382287), ...]


  >>> model.wv.doesnt_match("breakfast cereal dinner lunch".split())
  'cereal'

  >>> model.wv.similarity('woman', 'man')
  0.73723527

Probability of a text under the model::

  >>> model.score(["The fox jumped over a lazy dog".split()])
  0.2158356

Correlation with human opinion on word similarity::

  >>> model.wv.evaluate_word_pairs(os.path.join(module_path, 'test_data','wordsim353.tsv'))
  0.51, 0.62, 0.13

And on analogies::

  >>> model.wv.accuracy(os.path.join(module_path, 'test_data', 'questions-words.txt'))

and so on.

If you're finished training a model (i.e. no more updates, only querying),
then switch to the :mod:`gensim.models.KeyedVectors` instance in wv

  >>> word_vectors = model.wv
  >>> del model

to trim unneeded model memory = use much less RAM.

Note that there is a :mod:`gensim.models.phrases` module which lets you automatically
detect phrases longer than one word. Using phrases, you can learn a word2vec model
where "words" are actually multiword expressions, such as `new_york_times` or `financial_crisis`:

    >>> bigram_transformer = gensim.models.Phrases(sentences)
    >>> model = Word2Vec(bigram_transformer[sentences], size=100, ...)

.. [1] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean.
       Efficient Estimation of Word Representations in Vector Space. In Proceedings of Workshop at ICLR, 2013.
.. [2] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean.
       Distributed Representations of Words and Phrases and their Compositionality. In Proceedings of NIPS, 2013.
.. [3] Optimizing word2vec in gensim, http://radimrehurek.com/2013/09/word2vec-in-python-part-two-optimizing/
"""
#使用gensim中的word2vec模型 参考文档如上。
from gensim.models import Word2Vec

#word2vec的输入文档格式为 List<List<String>> 嵌套list，最小单元为 单词
f = open('./w2v/w2v.txt')
sentences = []
for line in f.readlines():
    sentences.append(line.split(" "))

model = Word2Vec(sentences,sg=1,size=200,window=5,min_count=5,negative=3,sample=0.001,hs=1,workers=4)
model.save('dbw2v.model')
