# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 16:43:40 2021

@author: mkkim
※requirements 
▶ python
▶ tensorflow
▶ opencv-python-4.5.3.56
▶ spyder 5 
▶ keras-2.6.0

"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb  17 13:17:50 2020

@author: Angelo
"""

import os

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from network import Network

import tensorflow.bitwise as tw 

class ICNET(Network):
    
    def __init__(self, cfg):
        
        self.batch_img = tf.compat.v1.placeholder(dtype=tf.uint8, shape=(None, None, None, 3))
        self.labels = tf.compat.v1.placeholder(dtype=tf.uint8, shape=(None,None,None,1))
        self.learning_rate = tf.compat.v1.placeholder(dtype=tf.float32)
        self.sum_losses = tf.compat.v1.placeholder(dtype=tf.float32)
        self.cfg = cfg
        self.reservoir = {}
        self.num_class = cfg.num_class
        self.ignore_label = 255
        
        #self.ignore_label
        
        super(ICNET, self).__init__()
    
 
      
    # HIGH BRANCH  - 2021-10-09 
    def _high_branch(self, inputs,name=None, reuse = tf.compat.v1.AUTO_REUSE):
        
        with tf.compat.v1.variable_scope(name, reuse=reuse):
        
            (self.feed(inputs)
                 .Convbnact((3,3,3,32),strides=[1,2,2,1],rate=1,name='cbn_1')
                 .Convbnact((3,3,32,32),strides=[1,2,2,1],rate=1,name='cbn_2')
                 .Convbnact((3,3,32,64),strides=[1,2,2,1],rate=1,name='cbn_3'))
                 
        return self.terminals[0] # (16, 90, 90, 64)


    def _mid_branch(self, inputs,name=None, reuse = tf.compat.v1.AUTO_REUSE):
        
        with tf.compat.v1.variable_scope(name, reuse=reuse):
            
            dw = tf.shape(inputs)[1:3]//2
            # ※ 주의: stride,rate의 defult값이 None이 아님 
            (self.feed(inputs)
                 .resize_bilinear(size=dw,name='down_nn1') # 이미지 사이즈가 홀수임 # (2, 45, 45, 64) 
                 .Convbnact((3,3,3,32),strides=[1,2,2,1],rate=1,name=name+'_cbn1')
                 .Convbnact((3,3,32,32),strides=[1,1,1,1],rate=1,name=name+'_cbn2')
                 .Convbnact((3,3,32,64),strides=[1,1,1,1],rate=1,name=name+'_cbn3')
                 .max_pool(pool_size=3,strides=2,name=name+'pool1')  # 여기까지 (2, 90, 90, 64)
                 .ResBlk(64,128,stride=1,rate=1,name='conv2_1')
                 .ResBlk(128,128,stride=1,rate=1,name='conv2_2')
                 .ResBlk(128,128,stride=1,rate=1,name='conv2_3')
                 .ResBlk(128,256,stride=2,rate=1,name='conv3_1')) #(2, 45, 45, 256)
            
            return self.terminals[0]  # (2, 45, 45, 256)
    
    
    def PyrmidPoolingModule(self,inputs,name=None,reuse = tf.compat.v1.AUTO_REUSE):
        
        with tf.compat.v1.variable_scope(name,reuse=tf.compat.v1.AUTO_REUSE):
            
            up_size=tf.shape(inputs)[1:3]
            side = inputs
            
            arr=[]
            split_nums=[1,2,3,6]
            for i in split_nums:
                (self.feed(inputs)
                     .Pooling(split_num=i,name='pool_1x1')
                     .conv_nn((1,1,1024,1024//4),strides=[1,1,1,1],rate=1,name='conv1x1')
                     .resize_bilinear(size=up_size,name='up_sampling'))
                
                arr.append(self.terminals[0])
            
            x_sub1 = tf.concat([arr[0],arr[1]],axis=3)
            x_sub2 = tf.concat([arr[2],arr[3]],axis=3)
            x = tf.concat([x_sub1,x_sub2],axis=3)
            
            x = tf.concat([side,x],axis=3)#2048 나오게 됨 
            
            (self.feed(x)
                 .conv_nn((1,1,2048,1024),strides=[1,1,1,1],rate=1,name='_conv1x1'))
            
            return self.terminals[0]
    

    # Low-Branch Architecture
    def _low_branch(self, inputs,name=None, reuse = tf.compat.v1.AUTO_REUSE):
        
        with tf.compat.v1.variable_scope(name, reuse=reuse):
            
            dw = tf.shape(inputs)[1:3]//2  # mid_branch결과값을 줄여서 월본사이즈 1/32로 줄여서 사용함
            #(2, 22, 22, 256)
            (self.feed(inputs)  
                 .resize_bilinear(size=dw,name='down_nn2') #(2, 22, 22, 256)
                 .ResBlk(256,256,stride=1,rate=1,name='conv3_2')
                 .ResBlk(256,256,stride=1,rate=1,name='conv3_3')
                 .ResBlk(256,256,stride=1,rate=1,name='conv3_4')
                 .ResBlk(256,512,stride=1,rate=2,name='conv4_1')
                 .ResBlk(512,512,stride=1,rate=4,name='conv4_2')
                 .ResBlk(512,512,stride=1,rate=6,name='conv4_3')
                 .ResBlk(512,512,stride=1,rate=2,name='conv4_4')
                 .ResBlk(512,512,stride=1,rate=4,name='conv4_5')
                 .ResBlk(512,512,stride=1,rate=6,name='conv4_6')
                 .ResBlk(512,1024,stride=1,rate=2,name='conv5_1')
                 .ResBlk(1024,1024,stride=1,rate=4,name='conv5_2')
                 .ResBlk(1024,1024,stride=1,rate=6,name='conv5_3'))  # (2, 22, 22, 1024)
            
            side = self.terminals[0]
            x = self.PyrmidPoolingModule(side,name='psp') #16,22,22,1024나와야함
    
            Sum = x + side
    
            (self.feed(Sum)
                 .Convbnact((1,1,1024,256),strides=[1,1,1,1],rate=1,name=name+'_cbn1'))
    
            return self.terminals[0]
        
        
    def _CascadeLabelGuidance(self,labels,name=None, reuse = tf.compat.v1.AUTO_REUSE):
        
        with tf.compat.v1.variable_scope(name,reuse=tf.compat.v1.AUTO_REUSE):
            dw_size = [16,8,4]
            arr =[]
            
            for dw in dw_size:
                sub='dw_size{}'.format(dw)
                size = tf.shape(labels)[1:3]//dw
            
                (self.feed(labels)
                     .resize_bilinear(size=size,name='upsample'))
                arr.append(self.terminals[0])
    
            return arr
        
        
    def CascadeFeatrueFusion(self,num1,num2,cin_num1,cin_num2,name=None, reuse=tf.compat.v1.AUTO_REUSE):
        
        with tf.compat.v1.variable_scope(name, reuse=reuse):
            up_size = tf.shape(num2)[1:3]

            F1_in = cin_num1  # 여기서 텐서가 넘파이로 안바뀌는 이유를 모르겠음 numpy() :x, eval(): x 
            F2_in = cin_num2
            
            #F1: (2, 45, 45, 128)
            (self.feed(num1)
                 .resize_bilinear(size=up_size,name='upsample'))
                 #.to_reservoir(name=name+'_loss')
            
            ########################################################################################
            side_left = self.terminals[0]             
            ########################################################################################
            side_right = self.terminals[0]
            ########################################################################################
            # loss way!
            ########################################################################################
            
            (self.feed(side_left)
                 .conv_nn((1,1,F1_in,19),strides=[1,1,1,1],rate=1,padding='SAME',name='conv_1x1'))
            
            side = self.terminals[0]             
          
            ########################################################################################
            # F2 way!
            ########################################################################################
            
            (self.feed(side_right)
                 .conv_nn((3,3,F1_in,128),strides=[1,1,1,1],rate=2,padding='SAME',name='conv_nn1')
                 .batch_normalization(name='_bn'))
            
            f1 = self.terminals[0]
        
            #F2: (2, 45, 45, 128)
            (self.feed(num2)
                 .conv_nn((1,1,F2_in,128),strides=[1,1,1,1],rate=1,padding='SAME',name='conv_ch')
                 .batch_normalization(name='_bn'))
            
            f2 = self.terminals[0]
            
            Sum = f1 + f2 
            
            (self.feed(Sum)
                 .activation(name='act')) #(2, 45, 45, 128)
            
        return side, self.terminals[0]
    
    def _build(self):
        
        ##########################################################
        # INFERENCE RESULTS
        ##########################################################
        
        inputs = tf.cast(self.batch_img, tf.float32)/255. - 0.5 # make input as float32 and in the range (-1, 1)
        self.reservoir['high_out'] = self._high_branch(inputs,name='high_nn')  # (2, 90, 90, 64)
        self.reservoir['mid_out'] = self._mid_branch(inputs ,name='mid_nn') #(2, 45, 45, 256)
        self.reservoir['low_out'] = self._low_branch(self.reservoir['mid_out'], name='low_nn') 
        
        # 21-10-25 유지보수해야할지점! 
        self.reservoir['pred1'], self.reservoir['F2'] = self.CascadeFeatrueFusion(self.reservoir['low_out'], self.reservoir['mid_out'],256,256, name='cfc')
        self.reservoir['pred2'],self.reservoir['F2_2'] = self.CascadeFeatrueFusion(self.reservoir['F2'], self.reservoir['high_out'],128,64, name='cfc2')
        
        
        up_size = tf.shape(self.reservoir['F2_2'])[1:3]*2
        
        (self.feed(self.reservoir['F2_2'])
             .resize_bilinear(size=up_size,name='upsample')
             .conv_nn((1,1,128,19),rate=1,strides=[1,1,1,1],name='resize'))
        
        self.reservoir['pred3'] = self.terminals[0]
        
        ##########################################################
        # label place hold shape
        ##########################################################
        labels = tf.cast(self.labels,tf.float32)
        self.reservoir['gt1'],self.reservoir['gt2'],self.reservoir['gt3'] =self._CascadeLabelGuidance(labels,name='cls')
        
        up_sizex4 = tf.shape(inputs)[1:3]
        
        ##########################################################
        # output visualization  resultion x1
        ##########################################################
        
        #(self.feed(self.reservoir['F2_2']) #출력채널이 128로 나와서 logit shape 720 720 128로 나옴  
        (self.feed(self.reservoir['pred3']) # 10-28 mk 수정 
             .resize_bilinear(size=up_sizex4,name='upsamplingx4'))
        
        self.reservoir['logits'] = self.terminals[0]  # shape: N*720*720*19가 되어야 함, 즉, total prediction! 
        

        ########################################################
        # NEEDED OPERATIONS FOR CALCULATING LOSS
        ########################################################
    
        self.loss_1 = self._createLoss(self.reservoir['pred1'], self.reservoir['gt1'],name='Loss1')
        self.loss_2 = self._createLoss(self.reservoir['pred2'], self.reservoir['gt2'],name='Loss2')
        self.loss_3 = self._createLoss(self.reservoir['pred3'], self.reservoir['gt3'],name='Loss3')
    
        self.losses = 0.4*self.loss_1 + 0.4*self.loss_2 + 1*self.loss_3
    
    
        ########################################################
        # METRIC : CUFUSE MATRIX,mIOU
        ########################################################
        
        #conf_mat = self._confusion_matrix(self.reservoir['logits'], labels)
        
        # 이걸 클래스별로 돌려야 값을 얻어와야함 
        
        #self.reservoir['mIoU'] = self.IoU_metric(conf_mat)
    
    
    
    def _createLoss(self,pred,gt, name=None, reuse=tf.compat.v1.AUTO_REUSE):
        
        ###################################################
        # LOSS CALCULATION
        ##################################################
        with tf.compat.v1.variable_scope(name, reuse=reuse):
            
            gt = tf.reshape(gt,(-1,)) # serialize
            mask = tf.less_equal(gt,self.num_class-1) # True wehn gt <=18
            indices = tf.squeeze(tf.where(mask),1)
            
            gt = tf.cast(tf.gather(gt, indices), tf.int32)
            pred = tf.gather(tf.reshape(pred, (-1,self.num_class)), indices)
            
            losses = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt, logits=pred))
        
            return losses 
             
        
    def optimizer(self,name=None, reuse=tf.compat.v1.AUTO_REUSE):
        ###############################################
        #  WEIGHT DECAY (exclude Batch Norm Params)
        ##############################################
        t_var = tf.compat.v1.trainable_variables()
        #w_var = [var for var in t_var if not('conv1x1' in var.name) or not('bn' in var.name) ]  # except for 1*1 convolutions
        w_var = [var for var in t_var if not('conv1x1' in var.name)]  # except for 1*1 convolutions
        w_l2 = tf.add_n([tf.nn.l2_loss(var) for var in w_var])
        loss_to_opt = self.losses + self.cfg.weight_decay * w_l2
                            
        ####################################################
        # OPTIMIZERS (i.e., Adam or RMSProp, etc) SETTING
        #####################################################
        
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.cfg.learning_rate, beta1=0.9, beta2=0.99)
        train_op = opt.minimize(loss_to_opt)


        #####################################################
        # CREATE SESSION
        #####################################################
        
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True, allocator_type='BFC')
        # when multiple GPUs - gpu_options = tf.GPUOptions(allow_growth=True, allocator_type='BFC', visible_device_list="0,1")
        config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        self.sess = tf.compat.v1.Session(config=config)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        
    
        #########################################################
        # CHECKPOINT-PROCESSING
        #########################################################
        self.saver = tf.compat.v1.train.Saver()
        ckpt_loc = self.cfg.ckpt_dir
        
        self.ckpt_name = os.path.join(ckpt_loc, 'ICNET')
        
        ckpt = tf.compat.v1.train.get_checkpoint_state(ckpt_loc)
        if ckpt and ckpt.model_checkpoint_path:
            import re
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.start_step = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print("---------------------------------------------------------")
            print(" Success to load checkpoint - {}".format(ckpt_name))
            print(" Session starts at step - {}".format(self.start_step))
            print("---------------------------------------------------------")
        else:
            if not os.path.exists(ckpt_loc):
                os.makedirs(ckpt_loc)
            self.start_step = 0
            print("**********************************************************")
            print("  [*] Failed to find a checkpoint - Start from the first")
            print(" Session starts at step - {}".format(self.start_step))
            print("**********************************************************")
  
    
        #################################################
        # SUMMARY AND SUMMARY WRITER
        #################################################
        wd_loss = tf.compat.v1.summary.scalar("WD_Loss", self.sum_losses[0])
        ce_loss = tf.compat.v1.summary.scalar("CE_Loss", self.sum_losses[1])

        
        self.summary_icnet = tf.compat.v1.summary.merge((wd_loss, ce_loss))    
        self.writer = tf.compat.v1.summary.FileWriter(self.cfg.log_dir, self.sess.graph)
        
        #return self.reservoir['high_out']  # (16, 90, 90, 64)
        #return self.reservoir['mid_out']  # (16, 45, 45, 256)
        #return self.reservoir['low_out']  # (16, 22, 22, 256)
        
        #return self.reservoir['pred1'], self.reservoir['F2'] # (16, 22, 22, 256)
        #(16, 45, 45, 19)
        #(16, 45, 45, 128)
        
        #return self.reservoir['pred3'], self.reservoir['F2_2'] # (16, 22, 22, 256)
        #(16, 90, 90, 19)
        #(16, 90, 90, 128)
        #return  self.reservoir['gt1'] (16, 45, 45, 1)
        #return self.losses  # 6.5669546
        
        return train_op, loss_to_opt ,self.losses ,self.reservoir['logits'],self.summary_icnet
        #return self.reservoir['F2_2']
        
    def save(self, global_step):
                
        self.saver.save(self.sess, self.ckpt_name, global_step)
        print('\n The checkpoint has been created, step: {}\n'.format(global_step))
        
        