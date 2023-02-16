#!/home/dpnkr/tensorflow/bin/python
# TENSORFLOW REGRESSION (with hidden nodes) ON PARKINSON DISEASE PATIENTS' DATASET
# -- regressing the parameter 'JITTER' over other remaining features
import tensorflow as tf
import scipy.io as scio
from sklearn import preprocessing as pp
import numpy as np
import math as math
import matplotlib.pyplot as plt

matFile = scio.loadmat('parkinsonsdat_for_regression.mat') # loading matrix saved by matlab 
dataInit = matFile.get('data')
dataInit = np.array(dataInit) # loading data as np.array object
##
rown = 0 
for i in range(0,990):
    if dataInit[i,0] > 36.0:
        rown = i
        break   

print rown
##
dataInit = pp.scale(dataInit) # standardizing data (along each column or feature)
regList = [1,2,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21] # list of regressors

data = dataInit[:,regList] # X - matrix of regressors
Yj = dataInit[:,6] # output column for 'jitter percentage' regressand
Ym = dataInit[:,4] # output column for 'motor UPDRS' regressand

dataShp = data.shape # Getting shape of regressor matrix
N = dataShp[0] # no. of patients
F = dataShp[1] # no. of features (regressors)

#--- initial settings
tf.set_random_seed(1900) # in order to get repeatable results

#--- selecting % of training samples vs % of test samples
Nsamples = rown
#Nsamples = int(math.floor(.8*N)) 

Nsamples_test = N-Nsamples  # uncomment for test

x=tf.placeholder(tf.float32,[Nsamples,F]) #input for train
x2=tf.placeholder(tf.float32,[Nsamples_test,F]) #input for test

t=tf.placeholder(tf.float32,[Nsamples,1])
t2=tf.placeholder(tf.float32,[Nsamples_test,1])#desired outputs 

wt1 = tf.placeholder(tf.float32,[F,18])
wt2 = tf.placeholder(tf.float32,[18,10])
wt3 = tf.placeholder(tf.float32,[10,1])
bt1 = tf.placeholder(tf.float32,[Nsamples_test,18])
bt2 = tf.placeholder(tf.float32,[Nsamples_test,10])
bt3 = tf.placeholder(tf.float32,[Nsamples_test,1])
#--- neural netw structure:
#layer1: with 18 hidden nodes
w1=tf.Variable(tf.random_normal(shape=[F,18], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights1"))
b1=tf.Variable(tf.random_normal(shape=[Nsamples,18], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases1"))

a1=tf.matmul(x,w1)+b1
at1=tf.matmul(x2,wt1)+bt1
z1=tf.nn.tanh(a1)
zt1=tf.nn.tanh(at1)
#layer2: with 10 hidden nodes
w2=tf.Variable(tf.random_normal(shape=[18,10], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights2"))
b2=tf.Variable(tf.random_normal(shape=[Nsamples,10], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases2"))

a2=tf.matmul(z1,w2)+b2
at2=tf.matmul(zt1,wt2)+bt2
z2=tf.nn.tanh(a2)
zt2=tf.nn.tanh(at2)
#layerOut
w3=tf.Variable(tf.random_normal(shape=[10,1], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights3"))
b3=tf.Variable(tf.random_normal(shape=[Nsamples,1], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases3"))

y=tf.matmul(z2,w3)+b3 # neural network output for train
yreg=tf.matmul(zt2,wt3)+bt3 # neural network output for test
#--- optimizer structure
cost=tf.reduce_sum(tf.squared_difference(y, t, name="objective_function"))#objective function
costt=tf.reduce_sum(tf.squared_difference(yreg, t2, name="objective_function_test"))#objective function for test
optim=tf.train.GradientDescentOptimizer(3.5e-4,name="GradientDescent")# use gradient descent in the trainig phase 
optim_op = optim.minimize(cost, var_list=[w1,w2,w3,b1,b2,b3])# minimize the objective function changing w1
#--- initialize
init=tf.global_variables_initializer()
#--- run the learning machine
sess = tf.Session()
sess.run(init) 

# choose random rows for train data
np.random.seed(77)
rndSamples = np.arange(0,N,1) 
np.random.shuffle(rndSamples) # shuffle array containing indices of single patient 
tr_samplst = rndSamples[0:Nsamples] # choose randomly 625 samples
tst_samplst = rndSamples[Nsamples:N].tolist()

# choose first N rows of data
#tr_samplst = np.arange(0,Nsamples,1).tolist()
#tst_samplst = np.arange(Nsamples,N,1).tolist()

# train data
xval = data[tr_samplst,:]
y_samp = Yj[tr_samplst]
tval = y_samp.reshape((len(y_samp),1))

# test data
xtst = data[tst_samplst,:]
y_sampt = Yj[tst_samplst]
ytst = y_sampt.reshape((len(y_sampt),1))

N_its = 5000
for i in range(N_its):
    # train
    train_data={x: xval, t: tval}
    sess.run(optim_op, feed_dict=train_data)
    if i % 1000 == 0:# print the intermediate result
        print(i,cost.eval(feed_dict=train_data,session=sess))
    
    # code for parameter optimization
    if i==N_its-1:
        print "No. Iteration for Gradient Optim: %d" % (i+1)
        print "Training error: %f" % cost.eval(feed_dict=train_data,session=sess) # for Nits = 20000, train error =  1e-6
                                                                                  # for Nits = 5000, train error = 0.132  
#--- obtaining parameters from trained model
w1_final = sess.run(w1)
w2_final = sess.run(w2)
w3_final = sess.run(w3)
b1_final = sess.run(b1)
b2_final = sess.run(b2)
b3_final = sess.run(b3)

# new parameters for testing
tmean = np.mean(b1_final)
tstd = np.std(b1_final)
b1_fin_test = np.random.normal(tmean, tstd, (Nsamples_test,18))
tmean = np.mean(b2_final)
tstd = np.std(b2_final)
b2_fin_test = np.random.normal(tmean, tstd, (Nsamples_test,10))
tmean = np.mean(b3_final)
tstd = np.std(b3_final)
b3_fin_test = np.random.normal(tmean, tstd, (Nsamples_test,1))

#---initializing and running test session
sess_test = tf.Session()
sess_test.run(init)


test_data={x2: xtst, t2: ytst, wt1: w1_final, wt2: w2_final, wt3: w3_final, bt1: b1_fin_test, bt2: b2_fin_test, bt3: b3_fin_test}
sess_test.run(costt, feed_dict=test_data)
print "Test error: %f" % costt.eval(feed_dict=test_data, session=sess_test) # for Nits=20000, test error = 225.9
                                                                            # for Nits=5000, test error = 224.81
#---plotting results
yval = y.eval(feed_dict=train_data,session=sess)
yreg = yreg.eval(feed_dict=test_data,session=sess_test)

fig1 = plt.figure()
ax1 = fig1.add_subplot(121)
plt.title('Y_train vs Yhat_train')
ax1.plot(tval,yval,'ro')
plt.xlim(-2,6)
plt.ylim(-2,6)
plt.xlabel('Train data')
plt.ylabel('Output of Regression')
plt.grid(which='major', axis='both')
ax2 = fig1.add_subplot(122)
plt.title('Y_test vs Yhat_test')
ax2.plot(ytst,yreg,'bo')
plt.xlim(-2,6)
plt.ylim(-2,6)
plt.xlabel('Test data')
plt.ylabel('Output of Regression')
plt.grid(which='major', axis='both')
plt.savefig('6_2_1.png')
plt.show()

#---closing session
sess.close()
sess_test.close()