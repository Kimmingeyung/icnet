"""
Created on Sat Mar. 09 15:09:17 2019

@author: mkkim

main for mnist

"""

import argparse,time ,utils
from utils import Config ,CityscapesReader, LRController
from model import ICNET
import numpy as np
import cv2

def get_arguments():
    
    parser = argparse.ArgumentParser('Implementation for MNIST handwritten digits 2020')
    
    parser.add_argument('--ld_epoch', 
                        type=int, 
                        default=10,  # 월랜 50
                        help='Learning Rate Linear Decay Start Epoch', 
                        required = False)    
    
    parser.add_argument('--ckpt_dir', 
                        type=str, 
                        default='./ckpt',
                        help='The directory where the checkpoint files are located', 
                        required = False)
    
    parser.add_argument('--log_dir', 
                        type=str, 
                        default='./logs',
                        help='The directory where the Training logs are located', 
                        required = False)
    
    parser.add_argument('--res_dir', 
                        type=str, 
                        default='./res',
                        help='The directory where the Training results are located', 
                        required = False)
    
    
    return parser.parse_args()

def _get_batch(buffer, lock):
    while(1):
        #print('Buffer Size: {:d}'.format(len(buffer)))# 16
        
        if len(buffer) == 0:
            time.sleep(0.1)
            
        else:
            lock.acquire()
            item=buffer.pop(0)
            lock.release()
            #print('Retrieved-Buffer Size {:d}'.format(len(buffer)))
            break
        
    return item
        
def main():
    
    args = get_arguments()
    cfg = Config(args)
    
    print("---------------------------------------------------------")
    print("         Starting Cityscapes-Data Batch Processing Example")
    print("---------------------------------------------------------")
    
    cityscapes = CityscapesReader(cfg)  # batch reader기 
    e_cityscapes = CityscapesReader(cfg,training=False)  # batch reader기 
    
    lrctrl = LRController(cfg) # decay 
   
    batch_buffer = cityscapes.buffer
    batch_lock = cityscapes.lock
    
    e_batch_buffer = e_cityscapes.buffer
    e_batch_lock = e_cityscapes.lock
    
    #images,labels = cityscapes._get_batch()
    #images,labels = _get_batch(batch_buffer,batch_lock)
    
    
    print('-======>model define')
    net = ICNET(cfg)
    
    _train_op , _wd_loss , _loss, _logits ,_summary_icnet = net.optimizer()
    

    global_step = net.start_step
    
    SAVE_STEP = 10
    
    #####################################################
    #        MAIN-TRAINING
    #####################################################            
    
    max_cost = 0.0
    
    per_epoch = cityscapes.img_list_size // cityscapes.batch_size  # 185
    e_iter = e_cityscapes.img_list_size // e_cityscapes.batch_size #500
    
    st_time = time.time()
    
    for epoch in range(global_step, cfg.num_epoch):
                       
        mean_cost = 0.
        mean_wd_cost = 0.
        curr_lr = lrctrl.get_lr(epoch)
   
        for step in range(per_epoch):
            
            images, labels = _get_batch(batch_buffer,batch_lock)
            
            feed_dict = {net.batch_img:images, net.labels:labels,net.learning_rate:curr_lr}           
            _ , wd_loss, loss = net.sess.run((_train_op, _wd_loss, _loss), feed_dict=feed_dict)
            
        
        # 1) loss 평균 출력 : o
            #print(wd_loss,loss)
            due = time.time() - st_time
            print("Learning at {:d} epoch :: WD_Cost - {:1.8f}, Cost - {:1.8f},time={}".format(step, wd_loss, loss,due))
            
            mean_cost += loss
            mean_wd_cost += wd_loss
               
        mean_cost /= float(per_epoch)
        mean_wd_cost /= float(per_epoch)
       
        print("Learning at {:d} epoch :: WD_Cost - {:1.8f}, Cost - {:1.8f}".format(epoch, mean_wd_cost, mean_cost))
        print("Learning at {:d} epoch :: Cost - {:1.8f}".format(epoch,mean_cost))

        global_step += 1   # 11-01 mk      

        elapsed = time.time() - st_time
        emin = elapsed//60
        esec = elapsed - emin*60
        print("(elapsed - %d min. %1.2f sec.)"%(emin, esec))
   
        # SAVE CHECKPOINT (FOR EVERY SAVE_STEP EPOCHS)
        '''
        if global_step % SAVE_STEP == 0:
            net.save(global_step)
        '''
        
        ##################################################################
        # EVALUATION
        #################################################################
        # 1 epoch 돌고나서 evaluttion!
        if (epoch+1) % 1 == 0: 
            
            for e_step in range(e_cityscapes.img_list_size):
                e_images, e_labels = _get_batch(e_batch_buffer,e_batch_lock)
                
                fd = {net.batch_img:e_images, net.labels:e_labels}
                logits = net.sess.run(_logits,feed_dict = fd)
                
                #print(e_images.shape) # (1, 720, 720, 3)
                #print(e_labels.shape) #(1, 720, 720, 1)
                #print(logits.shape) #(1, 720, 720, 19)
                
                pred = np.argmax(logits,axis=3)

                #print(pred.shape) # (1, 720, 720)
                #e_cityscapes.save_cityscapes(e_images, e_labels, pred , e_step,epoch)
                
                e_cityscapes.save_cityscapes(e_images, e_labels, pred , e_step,epoch)
                
            #print(_logits.shape)
        
     
        ###############################################################
        # Summary and Ceckpoint
        ################################################################
        feed_dict ={net.sum_losses:(mean_wd_cost, mean_cost)}
        
        summaries = net.sess.run(_summary_icnet, feed_dict=feed_dict)
        net.writer.add_summary(summaries, epoch)
    
        net.save(global_step)  # checkpoin가 저장? 
        # 3) checkpoint 처리
        '''
        if mean_cost < max_cost:
            net.save(global_step)  # 이걸해야 checkpoin가 저장이 됨 
            max_cost = mean_cost # 2021-11-01 여기 고침 Check point저장안됨 ;;;
       
        '''

       
if __name__ == '__main__':
       
    main() 
