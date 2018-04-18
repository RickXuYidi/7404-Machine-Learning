#coding=utf-8
'''
Created on 2018.04.09

@author: Xu Yidi
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#Define constants
hidden_nodes=10       #Hidden layer nodes
input_size=7      #Number of input features
output_size=1     #Number of output classes
lr=0.0006         #Learning rate
#——————————————————Import dataset——————————————————————
f=open('dataset.csv') 
df=pd.read_csv(f)
#Get colomns 2-9 for 7 features and 1 label
data=df.iloc[:,2:10].values  
#Show the shape of the imported data
print(np.shape(data))


#——————————————————Get train data——————————————————————
def get_train_data(batch_size=60,time_step=20,train_begin=0,train_end=5800):
    batch_index=[]
    data_train=data[train_begin:train_end]
    #Normalize the dataset
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  
    #Define the train set
    train_x,train_y=[],[]  
    for i in range(len(normalized_train_data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,:7]
       y=normalized_train_data[i:i+time_step,7,np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y


#——————————————————Get test data——————————————————————
def get_test_data(time_step=20,test_begin=5800):
    data_test=data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    #Normalize the dataset
    normalized_test_data=(data_test-mean)/std  
    #Define the size of the samples
    size=(len(normalized_test_data)+time_step-1)//time_step   
    test_x,test_y=[],[]  
    for i in range(size-1):
       x=normalized_test_data[i*time_step:(i+1)*time_step,:7]
       y=normalized_test_data[i*time_step:(i+1)*time_step,7]
       test_x.append(x.tolist())
       test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:,:7]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:,7]).tolist())
    return mean,std,test_x,test_y


#——————————————————Defining placeholders——————————————————————
#Weights and biases of appropriate shape to accomplish above task

weights={
         'in':tf.Variable(tf.random_normal([input_size,hidden_nodes])),
         'out':tf.Variable(tf.random_normal([hidden_nodes,1]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[hidden_nodes,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
       }

#——————————————————Defining the network——————————————————————
def lstm(X):     
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']  
    #Define the shape of the tensor that will be input 
    input=tf.reshape(X,[-1,input_size]) 
    input_rnn=tf.matmul(input,w_in)+b_in
    #Transform the tensor into 3 dimensions as the input of the LSTM cell
    input_rnn=tf.reshape(input_rnn,[-1,time_step,hidden_nodes])  
    cell=tf.nn.rnn_cell.BasicLSTMCell(hidden_nodes)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    #"Output_rnn" is the result of every output node of LSTM. "Final_states" is the result of the last cell.
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  
    #Define the input from the hidden layer to the output layer
    output=tf.reshape(output_rnn,[-1,hidden_nodes]) 
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states

#——————————————————train the model——————————————————
def train_and_test_lstm(batch_size=80,time_step=15,train_begin=2000,train_end=5800):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)
    pred,_=lstm(X)
    
    mean,std,test_x,test_y=get_test_data(time_step)
    
    #Loss function
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #Train for 1000 times
        for i in range(1001):
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            
            if i % 200==0:
                print("The loss of the "+str(i)+"th training is:",loss_)
                
#——————————————————Test the model——————————————————                
        test_predict=[]
        for step in range(len(test_x)-1):
          prob=sess.run(pred,feed_dict={X:[test_x[step]]})   
          predict=prob.reshape((-1))
          test_predict.extend(predict)
        test_y=np.array(test_y)*std[7]+mean[7]
        test_predict=np.array(test_predict)*std[7]+mean[7]
        #Get the average of the prediction accuracy
        acc=1-np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  
        print("\n\nThe accuracy of the prediction is: %.2f%%"%(acc * 100))
        #Plot the result
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b')
        plt.plot(list(range(len(test_y))), test_y,  color='r')
        plt.show()

train_and_test_lstm()