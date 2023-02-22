## Classification using Neural Networks

Classification is performed on arrhythmia patients’ database and implemented using neural networks from TensorFlow library. Results of classifiction are plotted and comments are made.

(*Clone this repository inside TensorFlow working environment before running the python code.*)

### Classification using 2 classes

Binary classification or classification using only two classes is performed to classify patients who either *‘have arrhythmia’* or *‘don’t have arrhythmia’*. A neural network consisting two hidden layers with layer 1 having 257 nodes and layer 2 having 128 nodes, and using *sigmoid function* as output activation function is implemented for this purpose. A model with following parameter is obtained,

- Training error: 1.01 
- Results match with doctors’ decision: **99.779**%

![image](https://user-images.githubusercontent.com/25234772/220733897-c4b13dfa-8062-4b5a-ae9c-68c29d1cd848.png)

### Classification using 16 classes

Classification using all sixteen classes, representing different levels of arrhythmia, is performed to classify patients. A neural network consisting two hidden layers with layer 1 having 64 nodes and layer 2 having 32 nodes, and using *softmax function* as output activation function is implemented for this purpose. A model with following parameter is obtained,

- Training error: 0.0089 
- Results match with doctors’ decision: **100.0**%

![image](https://user-images.githubusercontent.com/25234772/220734085-be556137-7af7-472d-94f2-afe422cfbfd9.png)

### Comments

Both neural networks have performed very well for classification. Results predicted by both networks matched with the decisions given by doctors for more than 99.5% of cases.

Neural network consisting of fewer hidden nodes and using softmax activation function performed slightly better than other network in spite of having fewer data points for each class to classify.
