# KNN Image Segmentation
Implementation of image segmentation using K-Nearest Neighbours(KNN) algorithm.

## Short Description
Here, KNN algorithm written from scratch has been used to do image segmentation/masking.

This might not be the best approach to do image segmentation. We just wanted to explore KNN for image segmentation and 
dit it. This is not intended for production purpose however exploration of KNN for image segmentation is encouraged.

## Requirements
We recommend using Python 3 and the implementation here is also done in Python 3 environment.
## Dependencies
- Numpy
- Pillow/PIL
## Installation
 Install Numpy and Pillow by executing the following commands in the terminal.
```
$ sudo apt-get install python-pip  
$ sudo pip install numpy scipy
$ sudo pip install pillow
```

## Instructions
* Run `./knn.py` with python3 as the interpreter(shebang would take care in our file) and the system would train the KNN
model.  After it has done the modelling, it visualizes the clusters.  We have implemented visualization in our use case
by displaying image mask.
* Developers can tune in the k parameter and apply any of the modifications as required.
## Further Enhancements
As naive KNN is not the best algorithm for image masking, there are many rooms improve.  Some of them are:
* Selection of initial centroids which account for potential best cluster regions.  We can use heuristic methods like K-means++.
Quality of clustering depends heavily on the set of intitial clusters.
* Intensity, texture, location, etc can be taken while computing distance rather than just RGB channel values.  We can also
take weighted sum of those parameters.