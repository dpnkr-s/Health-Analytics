## Regression with Neural Networks

Regression is performed on Parkinson’s disease patients’ database and it is implemented using neural networks from TensorFlow library. Results of regression are plotted and comparison is made between the performance of neural networks, with hidden nodes and without hidden nodes. Also, comparison is made between regression models obtained from neural networks and other methods implemented previously.

(Clone this repository inside TensorFlow working environment before running the python code.)

### NN w/o hidden nodes

Regression for feature (F0=7) or ‘total updrs’ (unified parkinson's disease rating scale) with squared error values of training and testing phase. 

- No. of iterations: 20,000
- Training error: 0.26
- Test error: 8.56

![image](https://user-images.githubusercontent.com/25234772/220730937-c6f516a0-cff1-4c13-802b-c2e69ea0139b.png)

### NN w/ hidden nodes

Regression for ‘total updrs’ feature using a neural network with two hidden layers with first layer containing 18 nodes and second layer containing 10 hidden nodes. *Tanh* is used as output activation function.

- No. of iterations: 5,000
- Training error: 0.132
- Test error: 224.81

![image](https://user-images.githubusercontent.com/25234772/220731812-89a2cf8c-3c85-49e4-b483-d9d6b6622cf7.png)

### Comments

Comparing between neural networks with and without hidden layers, it is observed that neural network with hidden layers achieve very low training error within fewer iterations as it fits to training data very fast.

However, this results in larger testing errors, thus neural network without hidden layers has better performance in terms of accurately predicting values of features in concern.
