#!/usr/bin/env python3
"""
Author: Suraj Regmi
Date: 21st November, 2018
Description: Implementation of naive KMeans algorithm to do simple image segmentation.
________________________________________________________________________________________________________________________

Note:
This might not be the best approach to do image segmentation. We have just explored K-means algorithm for image segmentation and nothing
more. This is not intended for production purpose however exploration of K-means clustering for image segmentation is encouraged.
"""

from PIL import Image

import numpy as np


class KMeans:

    def __init__(self, no_of_centroids, data):
        """
        :param no_of_centroids: no of centroids or clusters in data clustering
        :param data: data to be trained
        ________________________________________________________________________________________________________________

        Instance Variables:
        data_labels: array of labels of each data points
        masked_data: data having masked for of data points
        centroids: dictionary having centroids points and centroids indices
        """
        self.k = no_of_centroids
        self.centroids = dict()

        self.data = data
        self.data_labels = np.array([[None]*data.shape[1]]*data.shape[0])

        self.masked_data = None

    def set_random_centroids(self):
        """
        Sets random centroids in the beginning of training.

        *If the number of unique random centroids is less than the no of clusters intended, the random initialization
        is done again.
        """
        self.centroids = dict()
        self.centroids['positions'] = np.array(list(zip(
            np.random.randint(self.data.shape[0], size=self.k),
            np.random.randint(self.data.shape[1], size=self.k)))
        )
        self.centroids['values'] = self.data[self.centroids['positions'][:, 0], self.centroids['positions'][:, 1], :]

        if len(set(tuple(i) for i in self.centroids['values'])) != self.k:
            self.set_random_centroids()


    def get_best_label(self, i, j):
        """
        Returns the nearest centroid point from the data as given by indices i and j.
        :param i: index i
        :param j: index j
        :return: nearest centroid point index(refer to self.centroids)
        """
        point = self.data[i, j, :]
        distances = np.sqrt(np.sum((point - self.centroids['values'])**2, axis=1))
        return np.argmin(distances)

    def classify_data_points(self):
        """
        Classify each data point into cluster based on nearest distance.
        """
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                self.data_labels[i, j] = self.get_best_label(i, j)

    def generate_masked_data_points(self):
        """
        Generate the masked data points by using the data label and centroid values.
        """
        self.masked_data = np.zeros((self.data.shape[0], self.data.shape[1], 3), dtype='uint8')
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                self.masked_data[i, j, :] = self.centroids['values'][self.data_labels[i, j]]

    def display_masked_data_points(self, i):
        """
        Show the masked data points in an image*.
        *This is done for the image segmentation.  For different kind of problems, different visualization
        techniques may be implemented.

        :param i: iteration parameter
        """
        if i%1 == 0:
            Image.fromarray(self.masked_data).show()

    def tune_centroids(self):
        """
        Check if the centroids converge and if they do stop, otherwise set centroids as
        mean of the cluster data it belongs.

        :return: True if convergence has been achieved else None
        """
        temp = self.centroids['values'].copy()
        for i in range(self.k):
            class_data = self.data[self.data_labels == i]
            self.centroids['values'][i] = np.average(class_data, axis=0)

        if (temp == self.centroids['values']).all():
            return True

    def fit(self):
        """
        Fit the Kmeans model to the data given.
        """

        for _ in range(3):
            self.classify_data_points()
            stop_tuning = self.tune_centroids()
            if stop_tuning:
                break
