#!/usr/bin/env python3

"""
Implementation of k-means clustering on simple images to do image segmentation.

Note:
This might not be the best approach to do image segmentation. We have just explored K-means algorithm for image segmentation and nothing
more. This is not intended for production purpose however exploration of K-means clustering for image segmentation is encouraged.
"""

from PIL import Image

import numpy as np

from kmeans import KMeans


def preprocess_image(filename):
    """
    Preprocess image and return the image array(Numpy array).
    :param filename: path of file
    :return: numpy array of the image
    """
    image = Image.open(filename)
    image = image.resize((200, 200))
    image.show()
    return np.asarray(image)


def main():

    # get the numpy array of the image
    image_array = preprocess_image('images/sample_image_2.png')

    # take just three channels
    image_array = image_array[:, :, 0:3]

    # construct the kmeans model
    kmeans = KMeans(5, image_array)

    # set random centroids
    kmeans.set_random_centroids()

    # fit the model and visualize
    kmeans.fit()

    # generate the masked data points
    kmeans.generate_masked_data_points()

    # show the masked image
    kmeans.display_masked_data_points(1)


if __name__ == "__main__":
    main()