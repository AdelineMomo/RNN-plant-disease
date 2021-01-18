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
from InceptionV3_mutiOb_12_5 import VariableSequenceClassification
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
#        text_file = open(r"E:\Github\momo\data\dummy.txt", "r")
        lines = text_file.readlines()
        text_file.close()

        self._trainList = []  
        self._trainLabels = np.zeros(len(lines))
        source_dir = r'E:\Github\momo\color2'
#        source_dir = r'E:\Github\momo\color_onlineBingIPM'
        
        i = 0
        print("set up trainlist and trainlabel")
        while i < len(lines): 
            img_path,self._trainLabels[i] = lines[i].split(" ")
            a,b = img_path.split("/")
            img_path = os.path.join(source_dir,a,b)
            self._trainList.append(img_path)
            i = i + 1 
 
        
        text_file = open(r"E:\Github\momo\data\test.txt", "r")
#        text_file = open(r"E:\Github\momo\data_online\test.txt", "r")
#        text_file = open(r"E:\Github\momo\data\pepper_healthy.txt")
#        text_file = open(r"E:\Github\momo\data\dummy.txt", "r")
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


#KSIZE = 6
#STRIDES = 2
#DEPTH = 512
#num_hidden = 200
#max_length = 25



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

#vgg_checkpoint_path =  r"E:\Github\momo\tf_model\17052020\model\model_8240"
vgg_checkpoint_path = r"E:\Github\momo\tf_model\vgg_16.ckpt"
saver = tf.train.Saver(max_to_keep = None)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#model.vgg_saver.restore(sess, vgg_checkpoint_path)


saver.restore(sess, r"E:\Github\momo\tf_model\14112020\model\model_1270")


###############################################################################################
display_step = 100 # 20 epochs  = 16767 steps
display_step2 = 300# 500 # snapshot every 5 epochs

step = 1

while step * batch_size *n_minibatches < training_iters:     
            
            #if step > 4000:
            for i in range(n_minibatches):
                batch_x_conv, batch_y_di = plantclefdata.next_batch(batch_size)
                if i ==0:
                   sess.run(model.optimize_ZOR)
                sess.run(model.optimize_AOR, {data: batch_x_conv, batch_size2: batch_size, target_di: batch_y_di, dropout: dropTrain, is_training: True})
                
            _,train_summary = sess.run([model.optimize_TSR,summary_op], {data: batch_x_conv, batch_size2: batch_size, target_di: batch_y_di, dropout: dropTrain, is_training: True})

            #temp_layer = sess.run(model.end_points['Mixed_6b'] ,{data: batch_x_conv, batch_size2: batch_size, target_di: batch_y_di, target_crop: batch_y_crop, dropout: dropTrain, is_training: True})
            
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
  


###############################################################################################

#number_test_data = 10068 #180 #10068 # 63 #10068 # 180 #427 # 10068 
#
#test_list = plantclefdata.testList() 
#plantclefdata.Reset_index_in_epoch_test()
#
#sess = tf.Session()
#saver = tf.train.Saver(max_to_keep = None)
#model_complete_path =r"E:\Github\momo\tf_model\14112020\model\model_6300"
#saver.restore(sess,model_complete_path)
#
##true_txt = r'E:\Github\momo\kernel_8\bing_ipm\CNN_true.txt'
##false_txt = r'E:\Github\momo\kernel_8\bing_ipm\CNN_false.txt'
##
##with open(true_txt,'w') as fid:
##    pass
##
##with open(false_txt,'w') as fid:
##    pass
#
#num = 0
#correct = 0;
#while num < number_test_data:    
#        
#        print('Obs num {:7.0f}'.format(num))
#        test_data_conv, test_label_di = plantclefdata.PrepareTestingBatch(1) # step/epoch = 694.35 = All testing data tested        
#        test_rnn =  sess.run(model.prediction, {data: test_data_conv, batch_size2: 1, target_di: test_label_di, dropout: dropTest, is_training: False})
#        J = np.mean(test_rnn[1],0)
#        J1 = np.expand_dims(J, 0)
#        
#        test_logits =  sess.run(model.logits_cnn, {data: test_data_conv, batch_size2: 1, target_di: test_label_di, dropout: dropTest, is_training: False})
#        
#        J2 = (J1 + test_logits)/ 2
#        
#        output = np.argmax(J1,1)
#
#
#  
#        gt = np.argmax(test_label_di,1)
#        info_str = '%05i_%i_%i\n'%(num,output[0],gt[0])
#        print(info_str)
#        if (output==gt).astype(int) == 1:
#            correct = correct + 1
#            
##            with open(true_txt,'a') as fid:
##                fid.write(info_str)
##        else:
##            with open(false_txt,'a') as fid:
##                fid.write(info_str)            
#            
#        num = num+ 1                                
#                     
#
#
#top1= float(correct / number_test_data) * 100

################################################################################################################
#
#def load_image(img_path):
#  print("Loading image")
#  img = cv2.imread(img_path)
#  if img is None:
#      sys.stderr.write('Unable to load img: %s\n' % img_path)
#      sys.exit(1)
#  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#  return img
#
#
#number_test_data = 10068 #10068 #10068 # 180 #427 # 10068 
#test_list = plantclefdata.testList() 
#plantclefdata.Reset_index_in_epoch_test()
#
#sess = tf.Session()
#saver = tf.train.Saver(max_to_keep = None)
#model_complete_path =r"E:\Github\momo\tf_model\14112020\model\model_7837"
#saver.restore(sess,model_complete_path)
##output_path_all = r"E:\Github\momo\grad_cam_output_result"
#output_path_all = r"E:\Github\momo\kernel_8\grad_cam_output_result"
#
#smooth = True
#num = 0
#
#while num < number_test_data:
##    if num >= 7241:
#        output_path = os.path.join(output_path_all,'%05i'%num) 
#        if not os.path.exists(output_path):
#            os.makedirs(output_path)
#            
#        print('Obs num {:7.0f}'.format(num))
#        test_data_conv, test_label_di = plantclefdata.PrepareTestingBatch(1) # step/epoch = 694.35 = All testing data tested        
#
##        img = np.resize(test_data_conv,(size_VGG,size_VGG,3))
##        path1 = directory1 + "main.png"
##        scipy.misc.imsave(path1, img)
#        
#        
#        output, grads_val =  sess.run(model.grad_cam, {data: test_data_conv, batch_size2: 1, target_di: test_label_di, dropout: dropTest, is_training: False})
#        output = output[0]           # [10,10,2048]
#        grads_val = grads_val[0]	 # [10,10,2048]
#
#        weights = np.mean(grads_val, axis = (0, 1)) 			# [2048]
#        cam = np.ones(output.shape[0 : 2], dtype = np.float32)	# [10,10]
#        #cam = np.zeros(output.shape[0 : 2], dtype = np.float32)	# [10,10]
#
#        # Taking a weighted average
#        for i, w in enumerate(weights):
#            cam += w * output[:, :, i]
#
#        # Passing through ReLU
#        cam = np.maximum(cam, 0)
#        cam = cam / np.max(cam)
#        cam3 = cv2.resize(cam, (size_sync,size_sync))      
#        
##        cam3 = np.expand_dims(cam3, axis=2)
##        cam3 = np.tile(cam3,[1,1,3])
#    
#    
#      
#        with open(output_path + "/gradcam.pkl" ,'wb') as fid:
#            cPickle.dump(cam3,fid,protocol=cPickle.HIGHEST_PROTOCOL)
#        
#        img = load_image(test_list[num])
#        img = cv2.resize(img,(size_sync,size_sync))
#        
#        
#        with open(output_path + "/img.pkl" ,'wb') as fid:
#            cPickle.dump(img,fid,protocol=cPickle.HIGHEST_PROTOCOL)
#            
#
#        
#        
#        img = img.astype(float)
#        img /= img.max()
#            
#
##        cam3 = cv2.applyColorMap(np.uint8(255*cam3), cv2.COLORMAP_JET)
#        cam3 = cv2.cvtColor(np.uint8(255*cam3), cv2.COLOR_BGR2RGB)
#    
##         Superimposing the visualization with the image.
#        alpha =0.5 # 0.0025
#        new_img = img+alpha*cam3
#        new_img /= new_img.max()
#       
##        new_img =  img + 3*cam3
##        new_img /= new_img.max()
#         # Display and save
##        io.imshow(new_img)
##        plt.axis('off')
##        plt.savefig(output_path,bbox_inches='tight')
##        plt.show()
#    
#    
#        _,alpha_forward1,alpha_backward1= sess.run(model.alpha_list_com,{data: test_data_conv, batch_size2: 1, target_di: test_label_di, dropout: dropTest, is_training: False})
#        
#        alphas1 = np.array(alpha_forward1).swapaxes(1,0)
###        alphas2 = np.array(alpha_backward1).swapaxes(1,0)
#    
#        test_temp_list = sess.run(_channel_index(),{data: test_data_conv, batch_size2: 1, target_di: test_label_di, dropout: dropTest, is_training: False})
#        plt.figure(1) 
#
#        j = 0
#        while j < plantclefdata._max_seq:  
#
#            with open(output_path + '/T' + str(test_temp_list[j]) + ".pkl" ,'wb') as fid:
#            
##            with open(output_path + '/T' + str(j) + ".pkl" ,'wb') as fid:
#                cPickle.dump(alphas1[j,0,:].reshape(KSIZE,KSIZE),fid,protocol=cPickle.HIGHEST_PROTOCOL)
#            
#        
#        
#                      
##            if smooth:
###              alpha_img = skimage.transform.pyramid_expand(alphas1[j,0,:].reshape(w_width_slide,w_width_slide), upscale=16, sigma=20)
##              alpha_img = skimage.transform.resize(alphas1[j,0,:].reshape(KSIZE,KSIZE), [128, 128])
##            plt.imshow(alpha_img, alpha=0.7)
##            plt.set_cmap(cm.Greys_r)
##            plt.axis('off')
##            plt.savefig(path3)    
#
#
##        directory1 = savefigfile1 + img_name + "/"  
##        if not os.path.exists(directory1):
##           os.makedirs(directory1)
#    
##        directorytemp = savefigfile3 + img_name + "/"  
##        if not os.path.exists(directorytemp):
#        
#
#  
#        
##            plt.imshow(alpha_img, alpha=0.7)
#
##            path3 = directory1 + '/T' + str(int(test_temp_list[j])) + ".png"
##            path3 = directory1 + '/T' + str(j) + ".png"
##            scipy.misc.imsave(path3, alphas1[j,0,:].reshape(w_width_slide,w_height_slide))
#            
##            path4 = directory2 + '/T' + str(int(test_temp_list[j])) + ".png"
##            path4 = directory2 + '/T' + str(j) + ".png"
##            scipy.misc.imsave(path4, alphas2[j,0,:].reshape(w_width_slide,w_height_slide))
##            
##
#            j = j + 1
#
#
##           os.makedirs(directorytemp)
#           
##        directory2 = savefigfile2 +  img_name + "/"  
##        if not os.path.exists(directory2):
##             os.makedirs(directory2)
#        
##        img = np.resize(test_data_conv,(size_VGG,size_VGG,3))
##        path1 = directory1 + "main.png"
##        scipy.misc.imsave(path1, img)
##    
##        path2 = directory2 + "main.png"
##        scipy.misc.imsave(path2, img)
#    
#
#
#    
#    
#
#        num = num+ 1   

                                     
                     









































##########################################"""""
#number_test_data = 10068#211#41762# 10068#37727 #10068#37727 # 216 #10068 # 63 # 117 # 3 #216 #3# 63 # #216 # 216  ##211
##prob_path= '/home/suehan/Documents/dataset/PlantVIllage-Dataset/lmdb2/color-80-20Miss/analysis_feature_dataOnline_new/model20190222/test_ipm/prob/'
##prob_path= '/home/suehan/Documents/dataset/PlantVIllage-Dataset/lmdb2/color-80-20Miss/analysis_feature_new/model20190222/test/prob/'
#
#prob_path_di = '/home/suehan/Documents/dataset/PlantVIllage-Dataset/lmdb2/color-80-20_21Miss/analysis_feature_new/model20190423/test/prob_di/'
#prob_path_crop = '/home/suehan/Documents/dataset/PlantVIllage-Dataset/lmdb2/color-80-20_21Miss/analysis_feature_new/model20190423/test/prob_crop/'
#
##prob_path_di = '/home/suehan/Documents/dataset/PlantVIllage-Dataset/lmdb2/color-80-20_21Miss/analysis_feature_dataOnline_new/model20190423/test_ipm/prob_di/'
##prob_path_crop = '/home/suehan/Documents/dataset/PlantVIllage-Dataset/lmdb2/color-80-20_21Miss/analysis_feature_dataOnline_new/model20190423/test_ipm/prob_crop/'
#
#
#test_list = plantclefdata.testList() 
#plantclefdata.Reset_index_in_epoch_test()
#################################
#savefigfile1 ='/home/suehan/Documents/dataset/PlantVIllage-Dataset/lmdb2/color-80-20_21Miss/analysis_feature_new/model20190411/pepper_healthy/attn_visual_forward/'
#savefigfile2 ='/home/suehan/Documents/dataset/PlantVIllage-Dataset/lmdb2/color-80-20_21Miss/analysis_feature_new/model20190411/pepper_healthy/attn_visual_backward/'
##savefigfile3 ='/home/suehan/Documents/dataset/PlantVIllage-Dataset/lmdb2/color-80-20_21Miss/analysis_feature_new/model20190411/test/attn_visual_forward_temp/'
#
##savefigfile1 ='/home/suehan/Documents/dataset/PlantVIllage-Dataset/lmdb2/color-80-20_21Miss/analysis_feature_dataOnline_new/model20190411/test_ipm/attn_visual_forward/'
##savefigfile2 ='/home/suehan/Documents/dataset/PlantVIllage-Dataset/lmdb2/color-80-20_21Miss/analysis_feature_dataOnline_new/model20190411/test_ipm/attn_visual_backward/'
#number_alpha_patches =16
#
##stepSize = 35  
##w_width = 123
##w_height = 123
###for alpha act
#
##w_width_slide = 11
##w_height_slide = 11
#
#w_width_slide = 7
#w_height_slide = 7
##################################################################################################################
#
#
#sess = tf.InteractiveSession()
##model.restore_fn(sess)
#
#     
#    # Restore all the weights from the original CNN
#saver = tf.train.Saver(max_to_keep = None)
#saver.restore(sess, "/home/suehan/Documents/software/tensorboard_log/model20190423/model_8000")
# 
#num = 0
#smooth = True
#while num < number_test_data:    
#        
#        print('Obs num {:7.0f}'.format(num))
#        test_data_conv, test_label_di, test_label_crop, = plantclefdata.PrepareTestingBatch(1) # step/epoch = 694.35 = All testing data tested        
#        prob_di=  sess.run(model.prediction, {data: test_data_conv, batch_size2: 1, target_di: test_label_di, target_crop: test_label_crop, dropout: dropTest, is_training: False})
#        prob_crop,_ = sess.run(model.prediction_crop, {data: test_data_conv, batch_size2: 1, target_di: test_label_di, target_crop: test_label_crop, dropout: dropTest, is_training: False})
#
#        img_folder,img_name= test_list[num].split("/")
#        
#        output_file = prob_path_crop + img_name + '.mat'          
#        mdict = {}
#        mdict['prob'] = prob_crop
#        sio.savemat(output_file,mdict)
#        
#        pred_re = prob_di[0,:,:] 
#        output_file = prob_path_di + img_name + '.mat'          
#        mdict = {}
#        mdict['prob'] = pred_re
#        sio.savemat(output_file,mdict)
#        
#
##        _,_,alpha_forward1,alpha_backward1= sess.run(model.alpha_list_com, {data: test_data_conv, batch_size2: test_num_total, target_di: test_label_di, target_crop: test_label_crop, dropout: dropTest, is_training: False})#, batch_size: batch_size})
##       
##        alphas1 = np.array(alpha_forward1).swapaxes(1,0)
##        alphas2 = np.array(alpha_backward1).swapaxes(1,0)
##
##
##        directory1 = savefigfile1 + img_name + "/"  
##        if not os.path.exists(directory1):
##           os.makedirs(directory1)
##    
###        directorytemp = savefigfile3 + img_name + "/"  
###        if not os.path.exists(directorytemp):
###           os.makedirs(directorytemp)
##           
##        directory2 = savefigfile2 +  img_name + "/"  
##        if not os.path.exists(directory2):
##             os.makedirs(directory2)
##        
##        img = np.resize(test_data_conv,(size_VGG,size_VGG,3))
##        path1 = directory1 + "main.png"
##        scipy.misc.imsave(path1, img)
##    
##        path2 = directory2 + "main.png"
##        scipy.misc.imsave(path2, img)
##    
##        test_temp_list = sess.run(_channel_index(), feed_dict={data: test_data_conv, batch_size2: 1, target_di: test_label_di, target_crop: test_label_crop, dropout: dropTest, is_training: False})#, batch_size: batch_size})
##        plt.figure(1) 
##        j = 0
##        while j < number_alpha_patches:  
###            path3 = directorytemp + '/T' + str(int(test_temp_list[j])) + ".png"
####            scipy.misc.imsave(path3, alphas1[j,0,:].reshape(w_width_slide,w_height_slide))
###                      
###            if smooth:
####              alpha_img = skimage.transform.pyramid_expand(alphas1[j,0,:].reshape(w_width_slide,w_width_slide), upscale=16, sigma=20)
###              alpha_img = skimage.transform.resize(alphas1[j,0,:].reshape(w_width_slide,w_height_slide), [193, 193])
###            plt.imshow(alpha_img, alpha=0.7)
###            plt.set_cmap(cm.Greys_r)
###            plt.axis('off')
###            plt.savefig(path3)    
###        
##
##  
##        
###            plt.imshow(alpha_img, alpha=0.7)
##
##            path3 = directory1 + '/T' + str(int(test_temp_list[j])) + ".png"
###            path3 = directory1 + '/T' + str(j) + ".png"
##            scipy.misc.imsave(path3, alphas1[j,0,:].reshape(w_width_slide,w_height_slide))
##            
##            path4 = directory2 + '/T' + str(int(test_temp_list[j])) + ".png"
###            path4 = directory2 + '/T' + str(j) + ".png"
##            scipy.misc.imsave(path4, alphas2[j,0,:].reshape(w_width_slide,w_height_slide))
###            
###
##            j = j + 1
#
#
#
#        num = num+ 1                              
                       




