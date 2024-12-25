#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/12/20 12:39
# @Author  : Shutian
# @File    : embedding.py
# @Description    : embedding txt
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


class Embedding:
    def __init__(self, embedding_method):
        self.embedding_method = embedding_method

    def embedding_text(self, text_list, vector_size):
        if self.embedding_method == "tfidf":
            corpus_tfidf = np.asarray(TfidfVectorizer().fit_transform(text_list).todense())
            return corpus_tfidf
        elif self.embedding_method == "lsa":
            corpus_tfidf = TfidfVectorizer().fit_transform(text_list)
            lsa = make_pipeline(TruncatedSVD(n_components=vector_size), Normalizer(copy=False))
            corpus_lsa = lsa.fit_transform(corpus_tfidf)
            return corpus_lsa
        elif self.embedding_method == "lda":
            tf = CountVectorizer(min_df=1).fit_transform(text_list)
            lda = LatentDirichletAllocation(n_components=vector_size)
            corpus_lda = lda.fit_transform(tf)
            return corpus_lda
        else:
            raise Exception("embedding does not match!")
