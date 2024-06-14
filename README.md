# Leaf Image Classification Using Machine Learning

## Overview

This project aims to develop an effective leaf image classification system using machine learning algorithms. It includes extensive experimentation and evaluation of different models and hyperparameter tuning to identify the best-performing model. It has collected a diverse dataset consisting of 16 different datasets with 1,00,098 RGB images of 874 different categories of plant leaves. These images have been pre-processed using various techniques to ensure their quality and consistency.

It has utilized five classical machine learning algorithms, namely **K-Nearest Neighbors (KNN), Decision Tree (DT), Gaussian Naïve Bayes (GNB), Support Vector Machine (SVM), and Artificial Neural Network (ANN)**, for the classification of the leaf images. These algorithms were chosen due to their established performance in image classification tasks and their ability to handle large datasets.

This project has performed a comparative study of the performance of these classifiers using various evaluation metrics such as accuracy, precision, recall, and f1-score. The accuracy of each algorithm was evaluated based on its ability to correctly classify leaf images into their respective categories. It has employed pre-processing techniques to enhance the quality of the leaf images, including image resizing, normalization, and feature extraction.

### Data Augmentation

It has also employed a variety of data augmentation techniques to expand the dataset and enrich the diversity of the images. Through the utilization of this method, we were able to generate additional samples that retained similarities with the original images but introduced subtle variations. The augmented dataset, comprising 1,601,568 images spanning 874 categories of plant leaves, was then utilized for training and evaluating machine learning classifiers, resulting in improved classification performance. These measures were implemented to enhance the generalization capability of the classifiers and mitigate the risk of overfitting. Ultimately, this meticulous approach facilitated the attainment of the optimal classification performance for our plant leaves image dataset. This step was pivotal in ensuring that the classifiers learned from a diverse array of images and exhibited robustness in generalizing to new, unseen data. These techniques were applied to ensure that the classifiers received relevant and consistent input data for accurate classification.

The results of this project will provide valuable insights into the performance of different machine-learning algorithms for leaf image classification. The findings will help researchers and practitioners in the field of plant biology, agriculture, and environmental science to accurately classify plant leaves based on their images, which can have applications in plant species identification, disease detection, and ecosystem monitoring.

## Dependencies

* [Numpy](http://www.numpy.org)
* [Pandas](https://pandas.pydata.org)
* [OpenCV](https://opencv.org)
* [Matplotlib](https://matplotlib.org)
* [Scikit Learn](http://scikit-learn.org/)

It is recommended to use [Anaconda Python 3.10 distribution](https://www.anaconda.com) and using a `Jupyter Notebook`.
