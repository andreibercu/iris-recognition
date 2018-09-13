# Python Script for an Iris Recognition System using OpenCV

This Python application takes 2 infrared eye images as input and gives a matching score after comparing the 2 irises. If a matching score threshold is selected, it can decide if the images represent the same person.

The algorithm firstly localizes the iris region in the 2 eye images, identifies and encodes the keypoints characterizing each of the irises and then uses the SIFT algorithm provided by OpenCV to compare the 2 sets of key points. SIFT (Scale-invariant feature transform) is an algorithm used in computer vision to detect and describe local features in images.

For testing purposes I used an iris image database provided by CASIA (The Institute of Automation, Chinese Academy of Sciences) containing more than 16k eye images from more than 400 individuals. I generated approximately 50k test experiments and, if it is selected a matching score threshold so that the ‘false accept rate’ is 0 (no matches in comparisons between irises from 2 different persons), then the ‘false reject rate’ is approximately 25%, meaning the algorithm gives the right answer in 75% of cases when comparing iris images coming from the same person.
