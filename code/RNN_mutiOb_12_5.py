# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:01:02 2017

@author: root
"""
import math
import functools
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import vgg
from tensorflow.python.ops import rnn, array_ops
from tensorflow.contrib.rnn import GRUCell, DropoutWrapper, MultiRNNCell

import numpy as np


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class VariableSequenceClassification:

    def __init__(self, data, target, dropout, batch_size2, is_training, num_hidden=200, ctx_shape=[64,512]): #ctx_shape=[36,512]): #ctx_shape=[64,512]):  # default = 1.0
        self.class_di = 21

        self.KSIZE = 8
        self.STRIDES = 2
        self.max_length = 16
        self.length = 16

        self._istraining = is_training
        def _channel_shuffle(X, ksize, depth, groups=4): # groups=4): #5 #bqtch, 9,9,7500
            X = tf.reshape(X,[self.batch_size,-1,ksize*ksize*depth])  #  batch, 81, features 
            X = tf.transpose(X,[0,2,1]) # batch, features, 81 
            featuresR,_ = X.shape.as_list()[1:] # [features, 81 ]  81 = max_length
            in_channels_per_group = int((self.max_length)/groups)   #36/3 =12
            shape = tf.stack([-1, featuresR, groups, in_channels_per_group])  #batch,features,3,12
            X = tf.reshape(X, shape)
            X = tf.transpose(X, [0, 1, 3, 2]) #batch,features,12,3
            shape = tf.stack([-1, featuresR, self.max_length])  #batch,features,max_length
            X = tf.reshape(X, shape)
            return X


        self.img = data  #batch,512,14,14
        with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=0.0)):
            self.logits_cnn, self.end_points = vgg.vgg_16(self.img , num_classes=21, is_training=self._istraining,dropout_keep_prob=dropout)   #[batch_size, num_classes]
       
        self.L_new = slim.get_variables_to_restore()[-2:]
        
        self.L = slim.get_variables_to_restore()[0:-2]
        self.vgg_saver = tf.train.Saver(self.L)  
        
        self.data = self.end_points['vgg_16/conv5/conv5_3'] 
        

        
        self.conv_to_RNN = None
        self.ctx_shape = ctx_shape
        self.target = target
        self.batch_size = batch_size2
        self._num_hidden = num_hidden
        self._dropout = dropout
        self.DEPTH = ctx_shape[1]
        
        conv_patches = tf.extract_image_patches(images=self.data, ksizes=[1, self.KSIZE, self.KSIZE, 1], strides=[1, self.STRIDES, self.STRIDES, 1], rates=[1, 1, 1, 1], padding='VALID')  #batch, 6, 6, 7*7*768
        
#        res6 = _channel_shuffle(conv_patches, self.KSIZE, self.DEPTH) #batch,features,max_length
#        datatemp =  tf.transpose(res6,[0,2,1]) #batch,max_length, features
#        patches = tf.transpose(tf.reshape(datatemp, [self.batch_size,self.max_length, -1, self.DEPTH]), [0, 1, 3, 2])
#        self.dataT = tf.reshape(patches,[self.batch_size,-1,self.KSIZE*self.KSIZE*self.ctx_shape[1]]) 
#      
        self.dataT = tf.reshape(conv_patches,[self.batch_size,16,self.KSIZE*self.KSIZE*self.ctx_shape[1]]) 
        self.init_hidden_W = None
        self.init_hidden_b = None 
        self.alpha_list1 = None
        self.alpha_list1_re = None
        self.hidden_att_w1 = None
        self.hidden_att_b1 = None         
        self.image_att_w1 = None
        self.image_att_b1 = None                            
        self.att_w1 = None
        self.att_b1 = None      
        self.hidden_att_w_re1 = None
        self.hidden_att_b_re1 = None
        self.image_att_w_re1 = None
        self.image_att_b_re1 = None    
        self.att_w_re1 = None
        self.att_b_re1 = None        
        self.chanel_w = None
        self.chanel_b = None
        self.chanel_w_re =None
        self.chanel_b_re = None

        
        depth_multiplier=1.0
        min_depth=16
        if depth_multiplier <= 0:
             raise ValueError('depth_multiplier is not greater than zero.')
        self.depth = lambda d: max(int(d * depth_multiplier), min_depth)



        self.prediction
        self.error

        self.optimize_ZOR
        self.optimize_AOR
        self.optimize_TSR
        
        self.optimize_ZOC
        self.optimize_AOC
        self.optimize_TSC
        
        self.optimize_ZO
        self.optimize_AO
        self.optimize_TS
        self.alpha_list_com
        self.grad_cam

        


    @lazy_property        
    def prediction(self): 
        max_length_com = tf.shape(self.dataT)[1]

        with tf.variable_scope("ForwardGRU"):
           gru_cell_fw = GRUCell(self._num_hidden)
           gru_cell_fw = DropoutWrapper(gru_cell_fw, output_keep_prob=self._dropout)
           
           def cond(ind, h, output, alpha_list1):
               ml = self.max_length 
               return tf.less(ind, ml)
                  
           def body(ind, h, output, alpha_list1):
                 context = self.dataT[:,ind,:]   # take in the first sequence of all batches --> (batch, 512*14*!4)
                 context = tf.reshape(context,[-1, self.ctx_shape[1], self.ctx_shape[0]]) # (batch,512,196)
                 context = tf.transpose(context, [0, 2, 1]) #batch,196,512
                 context_flat = tf.reshape(context, [-1, self.ctx_shape[1]])  # (batch*196,512)                 

                 ##############################  attn1 ###########################              
                 h_attn = tf.matmul(h, self.hidden_att_w1) + self.hidden_att_b1         
                 context_encode = tf.matmul(context_flat, self.image_att_w1) + self.image_att_b1 
                 context_encode = tf.reshape(context_encode, [-1, self.ctx_shape[0], self.ctx_shape[1]]) # (batch,196,512)
                 context_encode = context_encode + tf.expand_dims(h_attn, 1) # (batch, 1, 512) -> + -> (batch, 196, 512)
                 context_encode = tf.nn.tanh(context_encode)

                 context_encode_flat = tf.reshape(context_encode, [-1, self.ctx_shape[1]]) # (batch_size*196, 512)
                 alpha = tf.matmul(context_encode_flat, self.att_w1) + self.att_b1  # (batch_size*196, 1)
                 alpha = tf.reshape(alpha, [-1, self.ctx_shape[0]]) # (batch_size, 196)
                 alpha = tf.nn.softmax(alpha) + 1e-10
                 weighted_context = tf.reduce_sum(context * tf.expand_dims(alpha, 2), 1) # (batch, 512)         
                          

                 h, _ = gru_cell_fw(weighted_context, h)
                 tf.get_variable_scope().reuse_variables()
                 alpha_list1 = alpha_list1.write(ind, tf.expand_dims(alpha, 1)) # (batch,1, 196)                 

                 
                 output = output.write(ind, tf.expand_dims(h, 1)) # (batch,1, 200)
                 ind += 1

                 return ind, h, output, alpha_list1
            
           ind = tf.constant(0) 
           context = self.dataT[:,ind,:]   # take in the first sequence of all batches --> (batch, 512*14*!4)
           context = tf.reshape(context,[-1, self.ctx_shape[1], self.ctx_shape[0]]) # (batch,512,196)
           context = tf.transpose(context, [0, 2, 1]) #batch,196,512
           
           h, self.init_hidden_W, self.init_hidden_b =self._linear2('init_hidden_W', tf.reduce_mean(context, 1), self._num_hidden, transferweight = 0, tantanh= True) # (batch,256)    
           initial_output = tf.TensorArray(dtype=tf.float32, size=self.max_length)
           initial_alpha_list1 = tf.TensorArray(dtype=tf.float32, size=self.max_length)

           ################################ weight init #####################
           self.hidden_att_w1, self.hidden_att_b1 = self._linear3('hidden_att_W1', self._num_hidden, self.ctx_shape[1], transferweight = 0) # (batch, 512)
           self.image_att_w1, self.image_att_b1 = self._linear3('image_att_W1', self.ctx_shape[1], self.ctx_shape[1], transferweight = 0) # using default initializer         
           self.att_w1, self.att_b1 = self._linear3('att_W1', self.ctx_shape[1], 1, transferweight = 0)

           ##########################################################################3
         
           t,_,output, alpha_list1= tf.while_loop(cond, body, [ind, h, initial_output, initial_alpha_list1], swap_memory=True)
           output_final = output.stack()            
           output_final = tf.reshape(output_final,[-1, self.batch_size, self._num_hidden])  # (max_seq,batch,200)
           output_final = tf.transpose(output_final, [1, 0, 2]) #batch,max_seq,200

           alpha_list1_final = alpha_list1.stack()    # (max_seq,batch,1,196)
           alpha_list1_final = tf.reshape(alpha_list1_final,[-1, self.batch_size, self.ctx_shape[0]])  # (max_seq,batch,196)
           alpha_list1_final = tf.transpose(alpha_list1_final, [1, 0, 2]) #batch,max_seq,196
           self.alpha_list1 = alpha_list1_final



##################################################### backward ###################################################################
        with tf.variable_scope("BackwardGRU"):
           gru_cell_re = GRUCell(self._num_hidden)
           gru_cell_re = DropoutWrapper(gru_cell_re, output_keep_prob=self._dropout)
           
           def length(dat):
               used = tf.sign(tf.reduce_max(tf.abs(dat), axis=2))
               length = tf.reduce_sum(used, axis=1)
               length = tf.cast(length, tf.int32)
               return length
           def cond_re(ind_re, h_re, output_re, alpha_list_re1):
               ml_re = self.max_length
               return tf.less(ind_re, ml_re)
            
          
           def body_re(ind_re, h_re, output_re, alpha_list_re1):
                 data_reverse =array_ops.reverse_sequence(
                     input=self.dataT, seq_lengths=length(self.dataT),
                     seq_dim=1, batch_dim=0)

                 context_re = data_reverse[:,ind_re,:]   # take in the first sequence of all batches --> (batch, 512*14*!4)
                 context_re = tf.reshape(context_re,[-1, self.ctx_shape[1], self.ctx_shape[0]]) # (batch,512,196)
                 context_re = tf.transpose(context_re, [0, 2, 1]) #batch,196,512
                 context_flat_re = tf.reshape(context_re, [-1, self.ctx_shape[1]])  # (batch*196,512)
                 ##############################  attn1 ###########################                  
                 h_attn_re = tf.matmul(h_re, self.hidden_att_w_re1) + self.hidden_att_b_re1
                 context_encode_re = tf.matmul(context_flat_re, self.image_att_w_re1) + self.image_att_b_re1
                 context_encode_re = tf.reshape(context_encode_re, [-1, self.ctx_shape[0], self.ctx_shape[1]]) # (batch,196,512)
                 context_encode_re = context_encode_re + tf.expand_dims(h_attn_re, 1) # (batch, 1, 512) -> + -> (batch, 196, 512)
                 context_encode_re = tf.nn.tanh(context_encode_re)
                         # compute alpha_ti --> evaluate per pixel info accross 512 maps
                 context_encode_flat_re = tf.reshape(context_encode_re, [-1, self.ctx_shape[1]]) # (batch_size*196, 512)
                 alpha_re = tf.matmul(context_encode_flat_re, self.att_w_re1) + self.att_b_re1   # (batch_size*196, 1)
                 alpha_re = tf.reshape(alpha_re, [-1, self.ctx_shape[0]]) # (batch_size, 196)
                 alpha_re = tf.nn.softmax(alpha_re) + 1e-10
                 weighted_context_re = tf.reduce_sum(context_re * tf.expand_dims(alpha_re, 2), 1) # (batch, 512) 
                                                   
                 
                 h_re, _ = gru_cell_re(weighted_context_re, h_re)
                 tf.get_variable_scope().reuse_variables()
                 alpha_list_re1 = alpha_list_re1.write(ind_re, tf.expand_dims(alpha_re, 1)) # (batch,1, 196)

                 output_re = output_re.write(ind_re, tf.expand_dims(h_re, 1)) # (batch,1, 200)

                
                 ind_re += 1                
                 
                 
                 return ind_re, h_re, output_re, alpha_list_re1

                 
                 
           ind_re = tf.constant(0)  
           data_reverse =array_ops.reverse_sequence(
               input=self.dataT, seq_lengths=length(self.dataT),
               seq_dim=1, batch_dim=0)

           context_re = data_reverse[:,ind_re,:]   # take in the first sequence of all batches --> (batch, 512*14*!4)
           context_re = tf.reshape(context_re,[-1, self.ctx_shape[1], self.ctx_shape[0]]) # (batch,512,196)
           context_re = tf.transpose(context_re, [0, 2, 1]) #batch,196,512
 

           h_re, self.init_hidden_W_re, self.init_hidden_b_re =self._linear2('init_hidden_W_re', tf.reduce_mean(context_re, 1), self._num_hidden, transferweight = 0, tantanh= True) # (batch,256)    
           initial_output_re = tf.TensorArray(dtype=tf.float32, size=self.max_length)
           initial_alpha_list_re1 = tf.TensorArray(dtype=tf.float32, size=self.max_length)

         
           ######################### weight initialisation #########################

           self.hidden_att_w_re1, self.hidden_att_b_re1 = self._linear3('hidden_att_W_re1', self._num_hidden, self.ctx_shape[1], transferweight = 0) # (batch, 512)
           self.image_att_w_re1, self.image_att_b_re1 = self._linear3('image_att_W_re1', self.ctx_shape[1], self.ctx_shape[1], transferweight = 0) # using default initializer         
           self.att_w_re1, self.att_b_re1 = self._linear3('att_W_re1', self.ctx_shape[1], 1, transferweight = 0) # (batch_size*196, 1)
           ####################################################################################################
           
           t_re,h_re,output_re, alpha_list_re1 = tf.while_loop(cond_re, body_re, [ind_re, h_re, initial_output_re, initial_alpha_list_re1], swap_memory=True)  # (max_seq,batch,1,1000)
           output_final_re = output_re.stack()
           output_final_re = tf.reshape(output_final_re,[-1, self.batch_size, self._num_hidden])  # (max_seq,batch,200)
           output_final_re = tf.transpose(output_final_re, [1, 0, 2]) #batch,max_seq,200
        
           alpha_list_final_re1 = alpha_list_re1.stack()    # (max_seq,batch,1,196)
           alpha_list_final_re1 = tf.reshape(alpha_list_final_re1,[-1, self.batch_size, self.ctx_shape[0]])  # (max_seq,batch,196)
           alpha_list_final_re1 = tf.transpose(alpha_list_final_re1, [1, 0, 2]) #batch,max_seq,196
        


           output_final_re2 = array_ops.reverse_sequence(
               input=output_final_re, seq_lengths=length(self.dataT),
               seq_dim=1, batch_dim=0)
          
           alpha_list_final_re21 = array_ops.reverse_sequence(
               input=alpha_list_final_re1, seq_lengths=length(self.dataT),
               seq_dim=1, batch_dim=0)
               

          
           self.alpha_list1_re = alpha_list_final_re21
           

     
     
        with tf.variable_scope("EmbedingGRU"):     

     
           output = tf.concat(axis=2, values=[output_final, output_final_re2])
           output = tf.reshape(output, [-1, 2*self._num_hidden])
           _temp, self.decode_class_W, self.decode_class_b  = self._linear2('decode_class_W', output, self.class_di, transferweight = 0) # (batch, 1000)

           prediction = tf.nn.softmax(_temp) + 1e-10
           prediction = tf.reshape(prediction, [-1, max_length_com, self.class_di])
    
           return prediction, _temp



    @lazy_property
    def alpha_list_com(self):
        pred = self.prediction
        alpha_forward1 = self.alpha_list1
        alpha_backward1 = self.alpha_list1_re
        return pred, alpha_forward1,alpha_backward1
        
    @lazy_property
    def cost(self):
        # Compute cross entropy for each frame.
        target = tf.expand_dims(self.target, 1)
        target = tf.tile(target, [1,self.max_length,1]) 
        targetX = tf.reshape(target, [-1, self.class_di])
        _, logits_di = self._prediction
        entropy_di = tf.nn.softmax_cross_entropy_with_logits(logits=logits_di, labels=targetX)   
        self.cross_entropy_di = tf.reduce_mean(entropy_di) 
        tf.summary.scalar("cost_rnn", self.cross_entropy_di) 


        entropy_cnn = tf.nn.softmax_cross_entropy_with_logits(logits= self.logits_cnn, labels=self.target)   
        self.cross_entropy_cnn = tf.reduce_mean(entropy_cnn) 
        tf.summary.scalar("cost_cnn", self.cross_entropy_cnn) 
     
     
        cross_entropy =   self.cross_entropy_di +  0 * self.cross_entropy_cnn 
        
        def add_hist(train_vars):
               for i in train_vars:
                   name = i.name.split(":")[0]
                   value = i.value()
                   tf.summary.histogram(name, value)
             
        train_vars = tf.trainable_variables()
        add_hist(train_vars)
         
        
        tf.losses.add_loss(cross_entropy)
        tf.summary.scalar("cost", cross_entropy)


        # Add L2 regularisation
        for var in tf.trainable_variables():
            with tf.name_scope("L2_regularisation/%s" % var.op.name):
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                     self._regulariser(var))

        #print([v.name for v in tf.trainable_variables()])
        
  
        return cross_entropy 
        
        

    @lazy_property
    def optimize_TSC(self):
        learning_rate = 0.0001 #0.001##0.001#
        tvs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="vgg_16")
        
        reg_losses = tf.add_n([ self._regulariser(v) for v in tvs ]) 
        with tf.control_dependencies([self.cost]):
            loss = self.cross_entropy_cnn
            total_loss = loss + reg_losses

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("regularisation_loss", reg_losses)
        tf.summary.scalar("total_loss", total_loss) 
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
           opt = tf.train.AdamOptimizer(learning_rate=learning_rate) 
           gvs = opt.compute_gradients(total_loss, tvs)
           return opt.apply_gradients([(self.accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])



    @lazy_property
    def optimize_TSR(self):
        learning_rate = 0.0001 #0.001##0.001#  
        tvs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="ForwardGRU") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="BackwardGRU") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="EmbedingGRU")
      
        reg_losses = tf.add_n([ self._regulariser(v) for v in tvs ])      
        with tf.control_dependencies([self.cost]):
            loss = self.cross_entropy_di
            total_loss = loss + reg_losses
        
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("regularisation_loss", reg_losses)
        tf.summary.scalar("total_loss", total_loss) 
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
           opt = tf.train.AdamOptimizer(learning_rate=learning_rate) 
           gvs = opt.compute_gradients(total_loss, tvs)
           return opt.apply_gradients([(self.accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])

    @lazy_property
    def optimize_TS(self):
        learning_rate = 0.0001 #0.001##0.001#

#     ################################################################################
        with tf.control_dependencies([self.cost]):
            loss = tf.add_n(tf.losses.get_losses())
            reg_losses = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            total_loss = loss + reg_losses
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("regularisation_loss", reg_losses)
        tf.summary.scalar("total_loss", total_loss) 
        tvs = tf.trainable_variables()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
           opt = tf.train.AdamOptimizer(learning_rate=learning_rate) 
           gvs = opt.compute_gradients(total_loss, tvs)
           return opt.apply_gradients([(self.accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])
       
        

    
    
         
       
        
    @lazy_property
    def optimize_AO(self):
    
        with tf.control_dependencies([self.cost]):
            loss = tf.add_n(tf.losses.get_losses())
            reg_losses = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            total_loss = loss + reg_losses
        tvs = tf.trainable_variables()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):   
           grads = tf.gradients(total_loss, tvs )
           return  [self.accum_vars[i].assign_add(gv) for i, gv in enumerate(grads)]



    @lazy_property
    def optimize_AOR(self):
        tvs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="ForwardGRU") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="BackwardGRU") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="EmbedingGRU")
        reg_losses = tf.add_n([ self._regulariser(v) for v in tvs ]) 
              
        with tf.control_dependencies([self.cost]):
            loss = tf.add_n(tf.losses.get_losses())
            total_loss = loss + reg_losses

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
             grads = tf.gradients(total_loss, tvs )
             return  [self.accum_vars[i].assign_add(gv) for i, gv in enumerate(grads)]


    @lazy_property
    def optimize_AOC(self):
        
        tvs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="vgg_16")      
        reg_losses = tf.add_n([ self._regulariser(v) for v in tvs ])
 
        
        with tf.control_dependencies([self.cost]):
            loss = tf.add_n(tf.losses.get_losses())
            total_loss = loss + reg_losses

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
           grads = tf.gradients(total_loss, tvs )
           return  [self.accum_vars[i].assign_add(gv) for i, gv in enumerate(grads)]

        
        

    @lazy_property
    def optimize_ZO(self):
        tvs = tf.trainable_variables()
        self.accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
        return [tv.assign(tf.zeros_like(tv)) for tv in self.accum_vars]

    @lazy_property
    def optimize_ZOR(self):
        tvs =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="ForwardGRU") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="BackwardGRU") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="EmbedingGRU")       
        self.accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
        return [tv.assign(tf.zeros_like(tv)) for tv in self.accum_vars]

    @lazy_property
    def optimize_ZOC(self):
        tvs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="vgg_16")
        self.accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
        return [tv.assign(tf.zeros_like(tv)) for tv in self.accum_vars]
    


    @lazy_property
    def error(self):
        target = tf.expand_dims(self.target, 1)
        target = tf.tile(target, [1,self.max_length,1]) 
        
        pred, _ = self.prediction
        mistakes = tf.equal(
            tf.argmax(target, 2), tf.argmax(pred, 2))
        mistakes = tf.cast(mistakes, tf.float32)  # true -> 1, false -> 0
        # Average over actual sequence lengths.
        mistakes = tf.reduce_sum(mistakes, axis=1) # shape = (batch_size,1)
        mistakes /= tf.cast(self.length, tf.float32)
        accuracy_di = tf.reduce_mean(mistakes)  # shape = (1,1)
        tf.summary.scalar("Accuracy_di", accuracy_di)

        pred_cnn=  tf.nn.softmax(self.logits_cnn, name='probability_cnn')
        mistakes_cnn = tf.equal(tf.argmax(self.target,1), tf.argmax(pred_cnn,1))
        mistakes_cnn = tf.cast(mistakes_cnn, tf.float32)  # true -> 1, false -> 0
        accuracy_cnn = tf.reduce_mean(mistakes_cnn)  # shape = (1,1)
        tf.summary.scalar("Accuracy_cnn", accuracy_cnn)  
            
        return accuracy_di, accuracy_cnn

    @lazy_property
    def grad_cam(self):
        # Conv layer tensor [?,10,10,2048]
        conv_layer = self.end_points['vgg_16/conv5/conv5_3'] 

        
        with tf.control_dependencies([self.cost]):
            loss = tf.add_n(tf.losses.get_losses())
            

        grads = tf.gradients(loss, conv_layer)[0]
        norm_grads = tf.divide(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))
        return conv_layer, norm_grads 


        
        
        
    @staticmethod
    def _weight_and_bias(in_size, out_size):
        ##weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        weight = tf.truncated_normal([in_size, out_size], stddev=1.0/math.sqrt(float(in_size*2)))
        #weight = tf.random_normal([2*in_size, out_size])
        
        ##bias = tf.constant(0.01, shape=[out_size])  #0.1
        bias = tf.zeros([out_size])
        #bias = tf.random_normal([out_size])
        
        return tf.Variable(weight), tf.Variable(bias)
 

    
    
    def _regulariser(self, var):
        """A `Tensor` -> `Tensor` function that applies L2 weight loss."""
        weight_decay = tf.multiply(tf.nn.l2_loss(var),
                                   0.00001, # 0.0001
                                   name="weight_loss")
        return weight_decay


    def _relu(self, tensor):
        """Helper to perform leaky / regular ReLU operation."""
        return tf.nn.relu(tensor)
    
    def _tanh(self, tensor):
        """Helper to perform leaky / regular ReLU operation."""
        return tf.nn.tanh(tensor)
        

    def _linear2(self, name, input_, output_dim, bias_init = 0, weight_init= 0, relu=False, transferweight = 0, tantanh = False,  trainable_choice=True):
        """
        Helper to perform linear map with optional ReLU activation.
    
        A weight decay is added to the weights variable only if one is specified.
        """
        
        input_dim = input_.get_shape()[1]
        with tf.variable_scope(name):
            
            if transferweight == 0:
               #weight_init = tf.truncated_normal_initializer(stddev=1.0/math.sqrt(float(input_dim*2)))
               weight_init = tf.truncated_normal_initializer(stddev=0.01)
               bias_init = 0.01 # 0.0 
            else:
               weight_init = tf.constant_initializer(weight_init) 
              
  
            weight = tf.get_variable(name="weight",
                                     shape=[input_dim, output_dim],
                                     dtype=tf.float32,
                                     initializer= weight_init,
                                     trainable=trainable_choice)
                         
            bias = tf.get_variable(
                         name="bias",
                         shape=output_dim,
                         dtype=tf.float32,
                         initializer=tf.constant_initializer(bias_init),
                         trainable=trainable_choice)          
                         
            output = tf.matmul(input_, weight) + bias 

            if relu: 
                output = self._relu(output)
                
            if tantanh: 
                output = self._tanh(output)                     
            return output, weight, bias 
          
    def _linear3(self, name, input_dim, output_dim, bias_init = 0, weight_init= 0, relu=False, transferweight = 0, tantanh = False,  trainable_choice=True):
        """
        Helper to perform linear map with optional ReLU activation.
    
        A weight decay is added to the weights variable only if one is specified.
        """
#        input_dim = input_.get_shape()[1]
        with tf.variable_scope(name):
            
            if transferweight == 0:
               #weight_init = tf.truncated_normal_initializer(stddev=1.0/math.sqrt(float(input_dim*2)))
               weight_init = tf.truncated_normal_initializer(stddev=0.01)
               bias_init = 0.01 # 0.0 
            else:
               weight_init = tf.constant_initializer(weight_init) 
              
  
            weight = tf.get_variable(name="weight",
                                     shape=[input_dim, output_dim],
                                     dtype=tf.float32,
                                     initializer= weight_init,
                                     trainable=trainable_choice)
                         
            bias = tf.get_variable(
                         name="bias",
                         shape=output_dim,
                         dtype=tf.float32,
                         initializer=tf.constant_initializer(bias_init),
                         trainable=trainable_choice)          
                         
            return weight, bias
    @staticmethod
    def _last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1] #int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant
        
