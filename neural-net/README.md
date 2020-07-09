# Computational-Intelligence

## Description of Project

In this project the packages of keras (tensorflow) and sci-kit learn are used.

This is a project about collaborative filtering for recommendation systems. 
Given the data set MovieLens 100K (https://grouplens.org/datasets/movielens/100k/) the project was all about creating a Neural Network that given a user as input produces as output predictions for all the movies (1682) of the data set.<br/>

The file [centering.py](centering.py) was created to center all the data by subtracting the mean of all ratings in every record of the data set. The result is having all the values with mean the 0 (zero) value so the classifier works better without being "stack" to positive values only.<br/>

Fistly, the data are transfered to a matrix. There the missing values are filled with the code [missing_values.py](missing_values.py)  in order to be given to the Neural Network later on.<br/>
The Network uses back-propagation with feed forward to learn the values and improve it's weights.<br/>
The input is given one-hot encoded, thus it is a diagonal sparse matrix 943X943.<br/>
The structure of the Neural Network is 943 input neuron, 1682 output neurons and 20 neurons in the hidden layer.<br/>
A lot of different were tested to decide which are the best values for the optimizer's parameters, the activation functions and the loss function.<br/>
The best combination was SGD as optimizer, sigmoid function in the hidden and linear in the output layer and mean squared error as loss function.<br/>
The metrics used are RMSE (Root Mean Squared Error) and MAE (Mean Average Error).<br/>

The code of the Neural Network is [NNpy](NN.py) file.

A Deep Neural Network is also created with 3 hidden layers to test it's performance compared to 1 hidden layer.<br/>
The code of the DNN is [DNN.py](DNN.py) file.

Much more detailed analysis lies in file [Report.pdf](Report.pdf)
