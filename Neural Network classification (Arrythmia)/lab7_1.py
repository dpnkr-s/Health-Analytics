#!/home/dpnkr/tensorflow/bin/python
# TENSORFLOW CLASSIFICATION ON ARRHYTHMIA PATIENTS' DATASET
# ---classification into two classes, using sigmoid activation function
import tensorflow as tf
import scipy.io as scio
from sklearn import preprocessing as pp
import numpy as np
import math as math
import matplotlib.pyplot as plt

matFile = scio.loadmat('dat.mat') # loading preprocessed and standardized data saved by matlab 
dataP = matFile.get('y')
outP = matFile.get('c')
data = np.array(dataP) # loading data as np.array object
out = np.array(outP)

dataShp = data.shape # Getting shape of regressor matrix
N = dataShp[0] # no. of patients
F = dataShp[1] # no. of features (regressors)

##--- initial settings
tf.set_random_seed(1900)#in order to get always the same results
#
Nsamples = N # as in this exercise we will only perform training phase  
#Nsamples_test = N-Nsamples  # uncomment for test
#
x=tf.placeholder(tf.float32,[Nsamples,F]) #input for train
#
t=tf.placeholder(tf.float32,[Nsamples,1])
#
##--- neural netw structure:
##layer1
nodes1 = 257
w1=tf.Variable(tf.random_normal(shape=[F,nodes1], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights1"))
b1=tf.Variable(tf.random_normal(shape=[Nsamples,nodes1], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases1"))
#
a1=tf.matmul(x,w1)+b1
z1=tf.nn.sigmoid(a1)

##layer2
nodes2 = 128
w2=tf.Variable(tf.random_normal(shape=[nodes1,nodes2], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights2"))
b2=tf.Variable(tf.random_normal(shape=[Nsamples,nodes2], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases2"))
#
a2=tf.matmul(z1,w2)+b2
z2=tf.nn.sigmoid(a2)

##layerOut
nodesOut = 1
w3=tf.Variable(tf.random_normal(shape=[nodes2,nodesOut], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights3"))
b3=tf.Variable(tf.random_normal(shape=[Nsamples,nodesOut], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases3"))
#
a3=tf.matmul(z2,w3)+b3
y=tf.nn.sigmoid(a3)
# neural network output for train

##--- optimizer structure
cost=tf.reduce_sum(tf.squared_difference(y, t, name="objective_function"))#objective function
optim=tf.train.GradientDescentOptimizer(1.5e-2,name="GradientDescent")# use gradient descent in the trainig phase (prev best= 1.7e-2)
optim_op = optim.minimize(cost, var_list=[w1,w2,w3,b1,b2,b3])# minimize the objective function changing w1

##--- initialize
init=tf.global_variables_initializer()
##--- run the learning machine
sess = tf.Session()
sess.run(init) 

## choose random rows for train data
#np.random.seed(77)
#rndSamples = np.arange(0,N,1) 
#np.random.shuffle(rndSamples) # shuffle array containing indices of single patient 
#tr_samplst = rndSamples[0:Nsamples] # choose randomly 625 samples
#tst_samplst = rndSamples[Nsamples:N].tolist()

## choose first N rows of data
##tr_samplst = np.arange(0,Nsamples,1).tolist()
##tst_samplst = np.arange(Nsamples,N,1).tolist()

## train data
xval = data
tval = out-1 # as sigmiid function gives output only as 0 or 1
# so its necessary to modify output vector (which contains values 1 or 2) 


N_its = 10000
for i in range(N_its):
    #train
    train_data={x: xval, t: tval}
    sess.run(optim_op, feed_dict=train_data)
    if i % 100 == 0:# print intermediate results
        print(i,cost.eval(feed_dict=train_data,session=sess)) # train error = 1.001 (almost same for Nits=25000)
    
#--- print final results
nn_class = y.eval(feed_dict=train_data,session=sess)
nn_class = np.round(nn_class)+1

count = 0
for i in range(0,len(nn_class)):
    if (nn_class[i] == out[i]):
        count += 1
        
print 'Results match: %f' % ((count/float(N))*100) # results match: 99.779% (same for Nits=25000)

#---plotting results
plt.plot(out,'ro',label='Actual values')
plt.plot(nn_class,'bx',label='Predicted values from model')
plt.ylim(0,3)
plt.xlabel('case number')
plt.grid(which='major', axis='both')
plt.legend()
plt.savefig('7_1.png')
plt.show()

#---closing session
sess.close()