from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from collections import defaultdict
import logging  # Setting up the loggings to monitor gensim

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=logging.INFO)


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.values())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                     for w in words if w in self.word2vec] or
                    [np.zeros(self.dim)], axis=0)
            for words in X
        ])


dir = '/home/szymon/juno/challenge/poleval/'

# w2vec_model_1 = Word2Vec.load(dir+"all-sentences-word2vec.model")
# w2vec_model_2 = Word2Vec.load(dir+"all-sentences-word2vec-m2.model")
w2vec_model_3 = Word2Vec.load(dir + "all-sentences-word2vec-m3.model")


w2vec_model_3.wv.cosine_similarities((w2vec_model_3.wv.get_vector("naukowy")+w2vec_model_3.wv.get_vector("dyscyplina"))/2, [w2vec_model_3.wv.get_vector("belgia"),w2vec_model_3.wv.get_vector("belgia")])

model = w2vec_model_3

w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}


corpus = ['This is the first document.','This document is the second document.','And this is the third one.','Is this the first document?']


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)


vectorized = TfidfEmbeddingVectorizer(w2v)

# d=w2vec_model_2[w2vec_model_2.wv.vocab]

i = 1
