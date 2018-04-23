TensorflowPractice
====
  This repository contains tf practice scripts

|Author|JYF|
|---|---|
|E-mail|jyf131462@163.com|

## Contents
* [MLP](#MLP)
* [FashionAI](#FashionAI)

## MLP
This code implements the basic mlp network for classification 
* train 
  - You need to run train.py. The script first runs data_loader to generate training data, then learns 
  the parameters. The result will be saved as checkpoint in models/ 
* test 
  - You need to run eval.py. The script generate testing data and visualize it. The classification result is like: 
  ![load failed](https://github.com/yfji/TensorflowPractice/blob/master/mlp.png "classification result") 

## FashionAI
This code implements the Fashion AI mission for clothes keypoint detection
