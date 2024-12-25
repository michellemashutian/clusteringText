#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/12/20 15:29
# @Author  : Shutian
# @File    : clustering.py
# @Description    : clustering txt
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, KMeans, MiniBatchKMeans
from sklearn import metrics


class Clustering:
    def __init__(self, clustering_method):
        self.clustering_method = clustering_method

    def clustering_text(self, text, threshold):
        if self.clustering_method == "Kmeans":
            labels, metric_list = self._kmeans(text, threshold)
            return labels, metric_list
        elif self.clustering_method == "minibatchKmeans":
            labels, metric_list = self._minibatch_kmeans(text, threshold)
            return labels, metric_list
        elif self.clustering_method == "AgglomerativeClustering":
            labels, metric_list = self._agglomerative_clustering(text, threshold)
            return labels, metric_list
        elif self.clustering_method == "dbscan":
            labels, metric_list = self._dbscan(text, threshold)
            return labels, metric_list
        elif self.clustering_method == "birch":
            labels, metric_list = self._birch(text, threshold)
            return labels, metric_list
        elif self.clustering_method == "AffinityPropagation":
            labels, metric_list = self._affinity_propagation(text, threshold)
            return labels, metric_list
        else:
            raise Exception("algorithm does not match!")

    @staticmethod
    def metric_score(text, labels):
        silhouette_score = metrics.silhouette_score(text, labels, metric='euclidean')
        calinski_score = metrics.calinski_harabasz_score(text, labels)
        davies_score = metrics.davies_bouldin_score(text, labels)
        return [silhouette_score, calinski_score, davies_score]

    def _kmeans(self, text, ncluster):
        cluster_res = KMeans(n_clusters=ncluster).fit(text)
        cluster_labels = cluster_res.labels_
        metric_list = self.metric_score(text, cluster_labels)
        return cluster_labels, metric_list

    def _minibatch_kmeans(self, text, ncluster):
        cluster_res = MiniBatchKMeans(n_clusters=ncluster).fit(text)
        cluster_labels = cluster_res.labels_
        metric_list = self.metric_score(text, cluster_labels)
        return cluster_labels, metric_list

    def _agglomerative_clustering(self, text, ncluster):
        cluster_res = AgglomerativeClustering(n_clusters=ncluster).fit(text)
        cluster_labels = cluster_res.labels_
        metric_list = self.metric_score(text, cluster_labels)
        return cluster_labels, metric_list

    def _dbscan(self, text, threshold):
        cluster_res = DBSCAN(eps=threshold).fit(text)
        cluster_labels = cluster_res.labels_
        metric_list = self.metric_score(text, cluster_labels)
        return cluster_labels, metric_list

    def _birch(self, text, threshold):
        cluster_res = Birch(threshold=threshold).fit(text)
        cluster_labels = cluster_res.labels_
        metric_list = self.metric_score(text, cluster_labels)
        return cluster_labels, metric_list

    def _affinity_propagation(self, text, threshold):
        cluster_res = AffinityPropagation(preference=threshold).fit(text)
        cluster_labels = cluster_res.labels_
        metric_list = self.metric_score(text, cluster_labels)
        return cluster_labels, metric_list
