# Face-Distance-Prediction

Machine Learning - Predict distance between the camera and your face  (only) from your own dataset using python and sklearn linear regression.

## THE GOAL

The goal is to detect **YOUR FACE** (raimu) :v  and **predict it's distance to your camera** by value of your face's height detected.

This project is created to make you understand and implement sklearn linear regression using single parameter.

## REQUIREMENTS

1. **Python**

   I am using python 3.6.4.
   You can download it here <https://www.python.org/downloads/>

2. **open-cv**

   To process the image like converting them to grayscale and etc. 
   run `pip install opencv-python`

3. **pandas**

   Known as next version of numpy, pandas is a structured arrays that provides methods for basic data structure. I use this to make it simpler to read the dataset created.

   run `pip install pandas`

4. **sklearn**

   A simple and efficient tools to perform Machine Learning in python.

   run `pip install sklearn`



## HOW TO DO IT

1. Run `FaceDetect.py` to build your dataset

   - set variable on line 11 to fit your distance

     `actual_distance = 30 #cm`

2. Run first step with different distances

   - I did it with 20, 30, 50 cm

3. Run `FaceImplement.py` and enjoy! ^_^
