#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 16:16:30 2019

@author: suehan
"""

import pickle as pkl


import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage
import skimage.transform

from PIL import Image
from skimage import io

import tensorflow as tf
import numpy as np
import os
import struct
import scipy.io as sio
from array import array as pyarray
from numpy import array, int8, uint8, zeros
import collections
import pickle
import scipy.misc

import functools
#import sets

from tensorflow.python.ops import rnn, array_ops
from tensorflow.contrib.rnn import GRUCell, DropoutWrapper, MultiRNNCell
from tensorflow.contrib import slim

import imageio
from RNN_mutiOb_12_5 import VariableSequenceClassification
from time import gmtime, strftime
from logging_util import makelog

from tensorflow.contrib.slim.nets import vgg
from datetime import datetime
import time

from six.moves import cPickle


import sys
sys.path.append(r"E:/Github/20190325_tensorflow_extract/models/research/slim")
from preprocessing import vgg_preprocessing


logfile=makelog()

class DataSet(object):


    def __init__(self):
        self.labeldi = 21 #36# 36#21
    
        """Construct a DataSet."""
        text_file = open(r"E:\Github\momo\data\train.txt", "r")

        lines = text_file.readlines()
        text_file.close()

        self._trainList = []  
        self._trainLabels = np.zeros(len(lines))
        source_dir = r'E:\Github\momo\color2'

        
        i = 0
        print("set up trainlist and trainlabel")
        while i < len(lines): 
            img_path,self._trainLabels[i] = lines[i].split(" ")
            a,b = img_path.split("/")
            img_path = os.path.join(source_dir,a,b)
            self._trainList.append(img_path)
            i = i + 1 
 
        
        text_file = open(r"E:\Github\momo\data\test.txt", "r")
        lines = text_file.readlines()
        text_file.close()


        self._testList = []  
        self._testLabels = np.zeros(len(lines))
      
        i = 0
        print("set up testlist and testlabel")
        while i < len(lines): 
            img_path,self._testLabels[i] = lines[i].split(" ")
            a,b = img_path.split("/")
            img_path = os.path.join(source_dir,a,b)
            self._testList.append(img_path)
            i = i + 1 


        self._num_examples = self._trainLabels.shape[0]
        self._perm_list = np.arange(self._num_examples)
        np.random.shuffle(self._perm_list)       
        self._trainLabelsPerm = self._trainLabels[self._perm_list]
 
        
        self._num_testexamples = self._testLabels.shape[0]
        self._perm_list_test = np.arange(self._num_testexamples)      
        
        self._batch_seq = 0      
        self._epochs_completed = 0
        self._index_in_epoch = 0  
        self._index_in_epoch_test = 0 
        self._max_seq = 16# 25 #16 #w = h = 14, stride = 4


        
        
    def index_in_epoch_test (self):
        return self._index_in_epoch_test 
    
    def trainList(self):
        return self._trainList
        
    def trainLabels(self):
        return self._trainLabels
    
        
    def trainLabelsPerm(self):
        return self._trainLabelsPerm
        
    def testList(self):
        return self._testList
        
    def testLabels(self):
        return self._testLabels
        
    
    def num_examples(self):
        return self._num_examples
        
    def num_testexamples(self):
        return self._num_testexamples
        
    def epochs_completed(self):
        return self._epochs_completed
        
    def index_in_epoch(self):
        return self._index_in_epoch
    
    def max_seq(self):
        return self._max_seq    
        
    def batch_seq(self):
        return self._batch_seq 
        

    def PrepareTrainingBatch(self, Newpermbatch, batch_size, indicator):
        
        batch_conv =[]
        if indicator == 1:
            i = 0
            while i < batch_size: 
              img_path = self._trainList[Newpermbatch[i]]
              batch_conv.append(img_path)
              i = i + 1
        else:
            i = 0
            while i < batch_size: 
               img_path = self._testList[Newpermbatch[i]]
               batch_conv.append(img_path)
               i = i + 1
               
        return batch_conv


        
    def dense_to_one_hot(self,labels_dense, num_classes=0):
        labels_dense = labels_dense.astype(int)
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        labels_one_hot = labels_one_hot.astype(np.float32)

        temp = zeros((labels_one_hot.shape[0],num_classes))
        i=0
        while i < labels_one_hot.shape[0]:
              temp[i] = labels_one_hot[i]
              i=i+1

        return temp


    def next_batch(self,batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            self._perm_list = np.arange(self._num_examples)
            np.random.shuffle(self._perm_list)
            #self._trainList = self._trainList[perm]
            self._trainLabelsPerm = self._trainLabels[self._perm_list]
            
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self.PrepareTrainingBatch(self._perm_list[start:end], batch_size, 1), self.dense_to_one_hot(self._trainLabelsPerm[start:end],self.labeldi)


        
    def PrepareTestingBatch(self,test_total):
        start = self._index_in_epoch_test
        self._index_in_epoch_test += test_total
        if self._index_in_epoch_test > self._num_testexamples:
            start = 0
            self._index_in_epoch_test = test_total
            assert test_total <= self._num_testexamples
        end = self._index_in_epoch_test
             
        return self.PrepareTrainingBatch(self._perm_list_test[start:end], test_total, 0), self.dense_to_one_hot(self._testLabels[start:end],self.labeldi)


        
     ## Testing   
    def Reset_index_in_epoch_test(self):   
        self._index_in_epoch_test = 0  
        
           
plantclefdata = DataSet()
####### Network Parameters ########       
training_iters = 754541 #
batch_size = 3# 15 # 15# 20#15  
n_minibatches =15 # 3 
test_num_total = 15# 35#15
num_classes_di = 21  ### number classes of output
dropTrain = 0.5 #0.5
dropTest = 1
size_VGG = 299
log_path = r"E:\Github\momo\tf_model\21112020\log"   #20190610
save_dir = r"E:\Github\momo\tf_model\21112020\model"


data = tf.placeholder(tf.string) 
target_di =  tf.placeholder("float", [None, num_classes_di])
dropout = tf.placeholder(tf.float32)
batch_size2 = tf.placeholder(tf.int32)
is_training = tf.placeholder_with_default(True, [])
weight_decay   = 0.00004   
size_sync = 224

######### RNN input param #####
KSIZE = 8
STRIDES = 2
DEPTH = 512
num_hidden = 200
max_length = 16


ctx_shape=[KSIZE*KSIZE,DEPTH]

def _proc_img(xi):
    
    image_file = tf.read_file(xi)
    image_decoded = tf.image.decode_jpeg(image_file, channels=3)
    perm_idx_tensor = tf.cast(image_decoded , tf.float32) #* (1. / 255)
    return vgg_preprocessing.preprocess_image(perm_idx_tensor,size_sync,size_sync,is_training = True)

    
fn_preproc = lambda xi:_proc_img(xi)

images = tf.map_fn(fn=fn_preproc,
              elems = data,
             dtype=np.float32)


def _channel_index():  #bqtch, 9,9,7500

         groups=4
         K = np.sqrt(max_length)
         K1 = int(K)
         a = np.arange(max_length).reshape((K1, K1,1))
         A = tf.convert_to_tensor(a, np.int32)
         
         
         A = tf.reshape(A,[1,-1,1])  #  batch, 81, features 
         A = tf.transpose(A,[0,2,1]) # batch, features, 81 
         featuresR,_ = A.shape.as_list()[1:] # [features, 81 ]  81 = max_length
         
         in_channels_per_group = int((max_length)/groups)   #36/3 =12        
         shape = tf.stack([-1, featuresR, groups, in_channels_per_group])  #batch,features,3,12
         A = tf.reshape(A, shape)
         A = tf.transpose(A, [0, 1, 3, 2]) #batch,features,12,3
         shape = tf.stack([-1, featuresR, max_length])  #batch,features,max_length
         A = tf.reshape(A, shape)
         A = tf.squeeze(A)
         return A
     


model = VariableSequenceClassification(images, target_di, dropout, batch_size2, is_training)  # default = 1.0


       
with tf.name_scope("tensorboard_writing"):
     summary_op = tf.summary.merge_all()
    
train_writer = tf.summary.FileWriter(log_path + '/train',
                                      graph=tf.get_default_graph())
test_writer = tf.summary.FileWriter(log_path + '/test')  


############################

vgg_checkpoint_path = r"E:\Github\momo\tf_model\vgg_16.ckpt"
saver = tf.train.Saver(max_to_keep = None)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver.restore(sess, r"E:\Github\momo\tf_model\14112020\model\model_1270")


###############################################################################################
display_step = 100 # 20 epochs  = 16767 steps
display_step2 = 300# 500 # snapshot every 5 epochs

step = 1

while step * batch_size *n_minibatches < training_iters:     
            

            for i in range(n_minibatches):
                batch_x_conv, batch_y_di = plantclefdata.next_batch(batch_size)
                if i ==0:
                   sess.run(model.optimize_ZOR)
                sess.run(model.optimize_AOR, {data: batch_x_conv, batch_size2: batch_size, target_di: batch_y_di, dropout: dropTrain, is_training: True})
                
            _,train_summary = sess.run([model.optimize_TSR,summary_op], {data: batch_x_conv, batch_size2: batch_size, target_di: batch_y_di, dropout: dropTrain, is_training: True})

            train_writer.add_summary(train_summary, (step * batch_size * n_minibatches))
            valid_loss = sess.run(model.cost,{data: batch_x_conv, batch_size2: batch_size, target_di: batch_y_di,  dropout: dropTrain, is_training: True})
            valid_acc_di, valid_acc_cnn  = sess.run(model.error, {data: batch_x_conv, batch_size2: batch_size, target_di: batch_y_di,  dropout: dropTrain, is_training: True})            

            strftime("%Y-%m-%d %H:%M:%S", gmtime())
            logfile.logging("Epoch" + str(step) + ", Validation Loss= " + \
                            "{:.6f}".format(valid_loss)  + ", Accuracy_cnn = " + \
                            "{:.5f}".format(valid_acc_cnn) + ", Accuracy_rnn = " + \
                            "{:.5f}".format(valid_acc_di))


            if step % display_step2 == 0:
                   saveid  = 'model_%s' %step
                   save_path = os.path.join(save_dir,saveid)
                   save_path =saver.save(sess, save_path)
                   
                   
            if step % display_step == 0:            
                   test_data_conv, test_label_di= plantclefdata.PrepareTestingBatch(test_num_total) # step/epoch = 694.35 = All testing data tested        
                   (test_acc_di, test_acc_cnn), test_summary =  sess.run([model.error, summary_op],{data: test_data_conv, batch_size2: test_num_total, target_di: test_label_di,  dropout: dropTest, is_training: False})                 
                   test_writer.add_summary(test_summary, (step * batch_size*n_minibatches))

                   logfile.logging("Epoch" + str(step) + ", Accuracy_cnn = " + \
                            "{:.5f}".format(test_acc_cnn * 100) + ", Accuracy_rnn = " + \
                            "{:.5f}".format(test_acc_di * 100))
                   
                  
            step += 1    
  














