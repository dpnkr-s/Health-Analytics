#!/home/dpnkr/tensorflow/bin/python
# TENSORFLOW REGRESSION (without hidden nodes) ON PARKINSON DISEASE PATIENTS' DATASET
# -- regressing the parameter 'MOTOR UPDRS' over other remaining features
import tensorflow as tf
import scipy.io as scio
from sklearn import preprocessing as pp
import numpy as np
import math as math
import matplotlib.pyplot as plt

matFile = scio.loadmat('parkinsonsdat_for_regression.mat')
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
dataInit = pp.scale(dataInit)
regList = [1,2,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21] #List of regressors

data = dataInit[:,regList] # X matrix of regressors
Yj = dataInit[:,6] # Output column for 'jitter percentage' regressand
Ym = dataInit[:,4] # Output column for 'motor UPDRS' regressand

dataShp = data.shape # Getting shape of regressor matrix
N = dataShp[0] # no. of patients
F = dataShp[1] # no. of features (regressors)

#--- initial settings
tf.set_random_seed(1900)#in order to get always the same results

#--- selecting % of training samples vs % of test samples
Nsamples = rown
# Nsamples = int(math.floor(.8*N)) 
# As the percentage of total samples used in training set increases, 
# test (prediction) error decreses even with more training error compared to 
# previous case with less training error but less % of trainnig samples

Nsamples_test = N-Nsamples  # uncomment for test

x=tf.placeholder(tf.float32,[Nsamples,F]) #input for train
x2=tf.placeholder(tf.float32,[Nsamples_test,F]) #input for test

t=tf.placeholder(tf.float32,[Nsamples,1])
t2=tf.placeholder(tf.float32,[Nsamples_test,1])#desired outputs 

wt = tf.placeholder(tf.float32,[F,1])
bt = tf.placeholder(tf.float32,[Nsamples_test,1])

#--- neural netw structure:
w1=tf.Variable(tf.random_normal(shape=[F,1], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights"))
b1=tf.Variable(tf.random_normal(shape=[Nsamples,1], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases"))
#bt=tf.Variable(tf.random_normal(shape=[Nsamples_test,1], mean=0.0, stddev=1.0, dtype=tf.float32, name="TestBiases"))

y=tf.matmul(x,w1)+b1 # neural network output for train
yreg=tf.matmul(x2,wt)+bt # neural network output for test
#--- optimizer structure
cost=tf.reduce_sum(tf.squared_difference(y, t, name="objective_function"))#objective function
costt=tf.reduce_sum(tf.squared_difference(yreg, t2, name="objective_function"))#objective function for test
optim=tf.train.GradientDescentOptimizer(0.9e-4,name="GradientDescent")# use gradient descent in the trainig phase
# eps = 1.462e-4 --> cost = 7e-6, costt = ~1090, overfitting??
# eps = 0.55e-4 --> cost = 1e-3, costt = ~1090, same again, not changing
optim_op = optim.minimize(cost, var_list=[w1,b1])# minimize the objective function changing w1,b1
#--- initialize
init=tf.global_variables_initializer()
#--- run the learning machine
sess = tf.Session()
sess.run(init)

# choose random rows for train data
np.random.seed(77)
rndSamples = np.arange(0,dataShp[0],1)
np.random.shuffle(rndSamples)
tr_samplst = rndSamples[0:Nsamples].tolist()
tst_samplst = rndSamples[Nsamples:dataShp[0]].tolist()

# choose first N rows of data
#tr_samplst = np.arange(0,Nsamples,1).tolist()
#tst_samplst = np.arange(Nsamples,N,1).tolist()

# train data
xval = data[tr_samplst,:]
y_samp = Ym[tr_samplst]
tval = y_samp.reshape((len(y_samp),1))

#test data
xtst = data[tst_samplst,:]
y_sampt = Ym[tst_samplst]
ytst = y_sampt.reshape((len(y_sampt),1))

err_tr = []
err_tst = []

N_its = 4500 # sweet spot found from bias vs variance inference curve
for i in range(N_its):
    # train
    train_data={x: xval, t: tval}
    sess.run(optim_op, feed_dict=train_data)
    
    # for parameter optimization
    if i==N_its-1:
        print "No. Iteration for Gradient Optim: %d" % (i+1)
        print "Training error: %f" % cost.eval(feed_dict=train_data,session=sess) # train error = 226.264
    #--- uncomment for finetuning and finding correct number of iterations for
    #--------------bias vs variance tradeoff (prevent overfitting and underfitting)
#==============================================================================
#    w1_final = sess.run(w1)
#    b1_final = sess.run(b1)
# 
#    tmean = np.mean(b1_final)
#    tstd = np.std(b1_final)
#    b1_fin_test = np.random.normal(tmean, tstd, (Nsamples_test,1))
#    #--- initialising variables again for testing
#    sess_test = tf.Session()
#    sess_test.run(init)
#
#    test_data={x2: xtst, t2: ytst, wt: w1_final, bt: b1_fin_test}
#    sess_test.run(costt, feed_dict=test_data)
#
#    if i % 100 == 0:# print the intermediate result
#        print(i,cost.eval(feed_dict=train_data,session=sess))
#        err_tr.append(cost.eval(feed_dict=train_data,session=sess))        
#        print (i,costt.eval(feed_dict=test_data, session=sess_test))
#        err_tst.append(costt.eval(feed_dict=test_data, session=sess_test))
    
#==============================================================================
w1_final = sess.run(w1)
b1_final = sess.run(b1)

    
tmean = np.mean(b1_final)
tstd = np.std(b1_final)
b1_fin_test = np.random.normal(tmean, tstd, (Nsamples_test,1))
#--- initialising and running test session
sess_test = tf.Session()
sess_test.run(init)

test_data={x2: xtst, t2: ytst, wt: w1_final, bt: b1_fin_test}
sess_test.run(costt, feed_dict=test_data)
print "Test error: %f" % costt.eval(feed_dict=test_data, session=sess_test) # test error = 227.752

#---plotting resuts

#plt.plot(err_tr,label='Training error')
#plt.plot(err_tst,label='Test error')
#plt.xlabel('Iterations')
#plt.grid(which='major',axis='both')
#plt.legend()
#plt.savefig('6_1_2_b.pdf',format='pdf')
#plt.show()
#
#plt.figure()
  
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
plt.savefig('6_1_2.png')
plt.show()

#---closing session
sess.close()
sess_test.close()