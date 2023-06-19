# -*- coding: utf-8 -*-
"""
Created on Sat Feb  22 15:17:50 2020

@author: Angelo
"""

import tensorflow as tf

DEFAULT_ACTIVATION = tf.nn.relu
DEFAULT_INITIALIZER = tf.initializers.he_normal()

def layer(op):
    '''Decorator for chaining components of layer'''
    def layer_decorated(self, *args, **kwargs):
        
        name = kwargs.setdefault('name', 'no_given_name')
        
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        
        layer_output = op(self, layer_input, *args, **kwargs)
        
        self.feed(layer_output)
        
        return self

    return layer_decorated

class Network(object):
    def __init__(self):

        # network terminal node
        self.terminals = []
        self.bn_pool = {}
        self._build()
        

    def _build(self, is_training):
        '''Construct network model. '''
        raise NotImplementedError('Must be implemented by the subclass in model.py')
        
    def feed(self, tensor):
        
        self.terminals = []
        self.terminals.append(tensor)
            
        return self
    

    @layer
    def conv_nn(self, inputs, filters, rate=1, strides=[1,1,1,1], padding='SAME',kernel_initializer=DEFAULT_INITIALIZER, name=None):
        
        with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
            
            kernels = tf.compat.v1.get_variable(name='kernel',shape=filters, initializer=kernel_initializer)
            
            if padding == 'REFLECT':
                
                pad1 = tf.cast(tf.subtract(filters[0:2], 1)/2, dtype=tf.int32)
                pad2 = tf.cast(tf.compat.v1.floordiv(filters[0:2], 2), dtype=tf.int32)
                pad_size = [[0, 0], [pad1[0],pad2[0]], [pad1[1], pad2[1]], [0,0]]
                inputs = tf.pad(inputs, pad_size, 'REFLECT')
                
                padding = 'VALID'
            
            x = tf.nn.conv2d(inputs, kernels, dilations=[1,rate,rate,1], strides=strides, padding=padding)
            
            return x
    

    @layer
    def reshape(self, inputs, shape, name=None):
        return tf.reshape(inputs, shape=shape, name=name)
    
    @layer
    def batch_normalization(self, inputs, name=None, reg='None', training=True):
        
        if (reg == 'None'):
            if name not in self.bn_pool.keys():
                self.bn_pool[name] = tf.keras.layers.BatchNormalization(name=name)
                
            output = self.bn_pool[name](inputs, training=training)
        else:
            output = inputs
    
        return output
    
    
    @layer
    def _batch_normalization(self,inputs,name=None,training=True):
        x = tf.keras.layers.BatchNormalization(name=name)(inputs,training=training)
        return x
    
    
    @layer
    def activation(self, inputs, name=None):
        return DEFAULT_ACTIVATION(inputs, name=name)
    
    @layer
    def Convbnact(self, inputs, filters, strides, rate, padding='SAME',kernel_initializer=DEFAULT_INITIALIZER, name=None):
        
        with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
            
            (self.feed(inputs)
                 .conv_nn(filters,strides=strides,rate=rate, padding=padding,name=name+'_conv1')
                 .batch_normalization(name=name+'_bn1')
                 .activation(name=name+'_act1'))
            
            return self.terminals[0]
            
    
    @layer
    def max_pool(self, inputs, pool_size=3, strides=2, padding='SAME', name=None):
        return tf.compat.v1.layers.max_pooling2d(inputs, pool_size, strides,padding=padding, name=name)
   
        
    @layer
    def avg_pool(self, inputs, pool_size=2, strides=2, padding='VALID', name=None):
        return tf.layers.average_pooling2d(inputs=inputs, pool_size=pool_size, strides=strides, name=name)
    
    
    @layer
    def resize_bilinear(self, inputs, size, name=None): # batch resize
        return tf.compat.v1.image.resize_bilinear(inputs, size=size, align_corners=True, name=name)
        
    @layer
    def resize_nn(self, inputs, size, name):
        return tf.compat.v1.image.resize_nearest_neighbor(inputs, size, name=name)  # label resize

    

    @layer 
    def ResBlk(self, inputs,c_in,c_out,stride=None,rate=None,name=None):
        ''' 레즈블락은 LONG타입만 정의함
        
        '''
        with tf.compat.v1.variable_scope(name, reuse = tf.compat.v1.AUTO_REUSE):
            
            #IDENTITY 
            if stride != 1:
                side =tf.nn.avg_pool2d(inputs, (stride,stride),(stride,stride),'SAME')
            else:
                side = inputs
                
           #PADDING
            if c_in != c_out:
                pad1 = (c_out-c_in)//2
                c_pad = [pad1, (c_out-c_in)-pad1]
                side = tf.pad(side, [[0,0], [0,0], [0,0], c_pad])
               
            #CONVOLUTION BLOCK
            
            c_mid = c_out//4
        
            (self.feed(inputs)
                 .Convbnact((1,1,c_in,c_mid),strides=[1,stride,stride,1],rate=1,name= name+'_cbn1')  # 첫번쨰 Conv만 stride지정
                 .Convbnact((3,3,c_mid,c_mid),strides=[1,1,1,1],rate=rate,name= name+'_cbn2') # 2번째 Conv만 rate 지정 
                 .conv_nn((1,1,c_mid,c_out),strides=[1,1,1,1],rate=1,name=name+'conv1') # 세번째 없음 
                 .batch_normalization(name= name +'_bn1'))
            
            x = self.terminals[0] + side # inputs값 즉, identity값 add 
    
            (self.feed(x)
                 .activation(name=name+'_act'))
        
            return self.terminals[0]    
    
    @layer
    def Pooling(self,inputs,split_num,name=None,reuse = tf.compat.v1.AUTO_REUSE):
        
        with tf.compat.v1.variable_scope(name,reuse=tf.compat.v1.AUTO_REUSE):
            # 10-23 유지보수해야함!!!!
            #split_num = 2 # number to be splited, i.e, 6x6
            
            h_size = tf.shape(inputs)[1]/split_num # horizontal unit size
            w_size = tf.shape(inputs)[2]/split_num # vertical unit size
            
            h=[tf.cast(h_size, tf.int32)] # first element
            w=[tf.cast(w_size, tf.int32)] # first element
            
            acc_h = h[0]
            acc_w = w[0]
            
            for idx in range(2,split_num+1):
                h.append(tf.cast(h_size*idx, tf.int32)-acc_h) # i.e., when split_num is 6, this gives 3, 4, 4, 3, 4, 4 splits
                acc_h += h[-1]
                
                w.append(tf.cast(w_size*idx, tf.int32)-acc_w)
                acc_w += w[-1]
                
                    
            h_sp = tf.split(inputs, h, axis=1) # horizontal split first
            
            h_merge = []
            for idx in range(len(h_sp)):
                w_sp = tf.split(h_sp[idx], w, axis=2) # vertical split
                h_merge.append(tf.concat([tf.reduce_mean(part, axis=(1,2), keepdims=True) for part in w_sp], axis=2)) 
                    
            
            h_merge = tf.concat(h_merge, axis=1) # concatenate the h_merge to be N x split_dim x split_dim x C
            
            return h_merge
    
    