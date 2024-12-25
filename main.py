#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/12/20 12:39
# @Author  : Shutian
# @File    : main.py
# @Description    :

from embedding import Embedding
from dataset import Dataset
from clustering import Clustering

if __name__ == "__main__":
    # filepath of the input text
    input_filename = "XXXX"
    # filepath of the stopwords
    stopword_filename = "XXXX"

    '''
    # "text", a two-column, tab-separated format, Text ID\tText content:
    text_id_1    This is the content of the first text.
    text_id_2    Here is another example of text content.

    # "word", a two-column, tab-separated format, Text ID\tWords(space-separated):
    text_id_1    keyword1 keyword2 keyword3
    text_id_2    another_keyword1 another_keyword2
    
    # "word_text", a three-column, tab-separated format, Text ID\tText content\tWords(space-separated):
    text_id_1    This is the content of the first text.    keyword1 keyword2 keyword3
    '''
    input_format = "text"

    # lsi, lda, vector_size needs to be set
    embedding_method = "tfidf"

    # "Kmeans", "minibatchKmeans", "AgglomerativeClustering"
    # "dbscan", "birch", eps or threshold needs to be set
    # "AffinityPropagation", preference needs to be set
    clustering_method = "Kmeans"

    dataset = Dataset(input_format)

    origin_data, text_data = dataset.load_dataset(filename=input_filename, stopword_filename=stopword_filename)

    embedding = Embedding(embedding_method)
    vec = embedding.embedding_text(text_list=text_data, vector_size=200)

    clustering = Clustering(clustering_method)
    ncluster = 10
    labels, output_metrics = clustering.clustering_text(text=vec, threshold=ncluster)

    # print [silhouette_score, calinski_score, davies_score]
    print(f"metrics of silhouette_score, calinski_score, davies_score : {output_metrics}")
    # text_label output
    result1 = open("clustering_output.txt", 'w', encoding="utf-8")
    for x, y in zip(origin_data, labels):
        result1.write(x + "\t" + str(y) + "\n")
    result1.close()
