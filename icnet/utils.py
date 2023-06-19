from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, cv2, glob, time, random
import numpy as np
from multiprocessing import Process, Manager, Lock

class Config(object):
    
    label_color = [[128, 64, 128], [244, 35, 232], [70, 70, 70]
                # 0 = road, 1 = sidewalk, 2 = building
                ,[102, 102, 156], [190, 153, 153], [153, 153, 153]
                # 3 = wall, 4 = fence, 5 = pole
                ,[250, 170, 30], [220, 220, 0], [107, 142, 35]
                # 6 = traffic light, 7 = traffic sign, 8 = vegetation
                ,[152, 251, 152], [70, 130, 180], [220, 20, 60]
                # 9 = terrain, 10 = sky, 11 = person
                ,[255, 0, 0], [0, 0, 142], [0, 0, 70]
                # 12 = rider, 13 = car, 14 = truck
                ,[0, 60, 100], [0, 80, 100], [0, 0, 230]
                # 15 = bus, 16 = train, 17 = motocycle
                ,[119, 10, 32]]
                # 18 = bicycle
    
    cityscape_data={
            'train_img_path': '.\\data\\leftImg8bit\\train',
            'train_label_path': '.\\data\\gtFine\\train',
            'test_img_path': '.\\data\\leftImg8bit\\val',
            'test_label_path': '.\\data\\gtFine\\val',
            'class_num': 19,
            'label_color': label_color
                }
    
    
    resize_low = 0.5
    resize_high = 2.0
    
    weight_decay = 0.0005
    
    batch_size = 16
    ebatch_size = 1
    
    num_epoch = 100  # 월래 100
    learning_rate = 0.001
    
    
    BUFFER_SIZE = 16
    TRAIN_SIZE = (720, 720)
    num_class = 19
            
    def __init__(self, args):
        print('Setup configurations...')
        
        self.ld_epoch = args.ld_epoch  # train epoch of 100 with linear decay after 50 epochs 
        
        self.ckpt_dir = args.ckpt_dir
        self.log_dir = args.log_dir
        self.res_dir = args.res_dir
        

class CityscapesReader(object):
    
    def __init__(self, cfg, training=True):
        
        self.training = training
        
        self.cfg = cfg
        self.label_colors = np.append(self.cfg.label_color, [[0, 0, 0]], axis=0)
        
        self.buffer = Manager().list([])
        
        
        self.buffer_size = cfg.BUFFER_SIZE
        
            
        self.lock = Lock()
        self.end_flag = Manager().list([False])
        
        if training:
            self.batch_size = cfg.batch_size
            #self.batch_size = 16
        else:
            self.batch_size = cfg.ebatch_size
            
        self.img_list = self._get_list()
        
        self.img_list_size = len(self.img_list)
        self.img_list_pos = 0
    
        self.res_dir = cfg.res_dir
                                    
        self.p = Process(target=self._start_buffer)
        self.p.daemon=True
        self.p.start()
        time.sleep(0.5)
    
    
    
    
    def _get_list(self):
        
        if self.training:
            train_cities = glob.glob(os.path.join(self.cfg.cityscape_data['train_img_path'], '*'))
            label_cities = glob.glob(os.path.join(self.cfg.cityscape_data['train_label_path'], '*'))
        else:
            train_cities = glob.glob(os.path.join(self.cfg.cityscape_data['test_img_path'], '*'))
            label_cities = glob.glob(os.path.join(self.cfg.cityscape_data['test_label_path'], '*'))
        
        train_cities.sort()
        label_cities.sort()
        
        img_list = []

        for idc, city in enumerate(train_cities):
            pngs = glob.glob(os.path.join(city, '*.png'))
                
            for idf, file in enumerate(pngs):
                fname = os.path.basename(file).split('_')
                fname[-1] = 'gtFine'
                fname += ['labelTrainIds.png']
                fname = '_'.join(fname)
                
                fname = os.path.join(label_cities[idc], fname)
                
                if os.path.exists(fname):
                    img_list.append((file, fname))
                else:
                    sys.exit('No matched - ', fname)

        print('List of (train_image, train_label) of ', len(img_list), '.... processing')                    
        return img_list
    
    
    def _start_buffer(self):

        while(1):
            
            if self.end_flag[0]:
                break
            
            _batch = self._get_batch()
                        
            while(1):
                if len(self.buffer) < self.buffer_size:
                    break
                else:
                    if self.end_flag[0]:
                        break
                    time.sleep(0.1)
                    
            self.lock.acquire()
            self.buffer.append(_batch)
            self.lock.release()
            #print('Stuffed - Buffer Size  {:d}'.format(len(self.buffer)))
            
    def _get_batch(self):
        
        if self.img_list_pos + self.batch_size > self.img_list_size-1:
            self.img_list_pos = 0
            random.shuffle(self.img_list)
        
        tr_cache = []
        lab_cache = []
                
        for index in range(self.batch_size):
            
            tr, lab = self._read_image(self.img_list[self.img_list_pos])
            
            tr_cache.append(tr)
            lab_cache.append(lab)
            
            self.img_list_pos += 1
                
        tr_batch = np.stack(tr_cache, axis=0)
        lab_batch = np.stack(lab_cache, axis=0)
       
        return (tr_batch, lab_batch)
    
    def _read_image(self, path):
        
        img_sample = cv2.imread(path[0], cv2.IMREAD_UNCHANGED)  # (1024, 2048, 3)
        lab_sample = cv2.imread(path[1], cv2.IMREAD_UNCHANGED)  # (1024, 2048)
                
        tr_size = np.array(self.cfg.TRAIN_SIZE) #array([720, 720])
                
        if self.training:
            # image augmentation - horizontal flip, and resizing
            
            flip = np.random.randint(1, 100) % 2
            if flip:
                img_sample = np.fliplr(img_sample)
                lab_sample = np.fliplr(lab_sample)
            
            resize_low = np.maximum(np.max(np.array(self.cfg.TRAIN_SIZE)/np.shape(img_sample)[:2]), self.cfg.resize_low)
            print('RESIZE LOW is - ', resize_low)
            ratio = np.random.uniform(low=resize_low, high=self.cfg.resize_high)
            new_size = np.flip(np.maximum(np.round(np.shape(img_sample)[:2] * np.array(ratio, dtype=np.float32)), tr_size)).astype(np.int32)
                        
            img_sample = cv2.resize(img_sample, tuple(new_size), interpolation=cv2.INTER_LANCZOS4) # (870, 1740, 3)
            lab_sample = cv2.resize(lab_sample, tuple(new_size), interpolation=cv2.INTER_NEAREST) # (870, 1740) # 여기도 nearest로 자름 
                        
            crop_pos = np.round((np.shape(img_sample)[0:2] - tr_size)*np.random.uniform()).astype(np.int32)
            
            if len(np.shape(lab_sample)) == 2:
                lab_sample = np.expand_dims(lab_sample, axis=2)
                
            img_sample = img_sample[crop_pos[0]:crop_pos[0]+tr_size[0], crop_pos[1]:crop_pos[1]+tr_size[1], :]  # 자르는 시작점 좌표값을 램덤으로 할당함
            lab_sample = lab_sample[crop_pos[0]:crop_pos[0]+tr_size[0], crop_pos[1]:crop_pos[1]+tr_size[1], :]
            
        else:
            #crop_pos = np.round((np.shape(img_sample) - tr_size)*np.random.uniform()).astype(np.int32)
            crop_pos = np.round((np.shape(img_sample)[0:2] - tr_size)*np.random.uniform()).astype(np.int32)  #10-28 mk수정
            
            if len(np.shape(lab_sample)) == 2:
                lab_sample = np.expand_dims(lab_sample, axis=2)
            
            img_sample = img_sample[crop_pos[0]:crop_pos[0]+tr_size[0], crop_pos[1]:crop_pos[1]+tr_size[1], :]
            lab_sample = lab_sample[crop_pos[0]:crop_pos[0]+tr_size[0], crop_pos[1]:crop_pos[1]+tr_size[1], :]
                            
        return (img_sample, lab_sample)
    
    
    def next_batch(self):
    
        while(1):
            
            if len(self.buffer) == 0:
                time.sleep(0.1)
                
            else:
                self.lock.acquire()
                item = self.buffer.pop(0)
                self.lock.release()
                #print('Retrieved - Buffer Size  {:d}'.format(len(bbuffer)))
                break
    
        return item
    
    def close(self):
        
        self.end_flag[0] = True
        print('Closing Processes....................................')
        time.sleep(1)
        
        
    def show_cityscapes(self, images, labels):
    
        sq_row = int(np.sqrt(np.shape(images)[0]))
            
        total_image = []
        total_label = []
                
        for row in range(sq_row):
            row_img = [images[id + row*sq_row] for id in range(sq_row)]
            row_lab = [labels[id + row*sq_row] for id in range(sq_row)]
            
            total_image.append(np.concatenate(row_img, axis=1))
            total_label.append(np.concatenate(row_lab, axis=1))
            
        show_img = np.concatenate(total_image, axis=0)
        show_lab = np.concatenate(total_label, axis=0)
        
        h, w = np.shape(show_lab)[:2]
        
        show_lab = np.where(show_lab==255, 19, show_lab)
        num_classes = 20
        
        index = np.reshape(show_lab, (-1,))
        
        one_hot = np.eye(num_classes)[index]
        show_lab = np.reshape(np.matmul(one_hot, self.label_colors), (h, w, 3))
        
        cv2.imshow('Training Image', show_img)
        cv2.imshow('Label Image', show_lab.astype(np.uint8))
        key = cv2.waitKey(0)
        
        return key
    
    def save_cityscapes(self, images, labels,pred, e_step,epoch):
    
        num_classes = 20

        images = np.squeeze(images) # (720, 720, 3)
        pred = np.squeeze(pred) # (720, 720)
        labels = np.squeeze(labels) # (720, 720)
        
        #출력을 위해서 월래 shape 복원할것 
        h,w = np.shape(pred)[0:2]  # 어차피 이건 interger 이므로 별도선언 생략하겠음
        pred = np.where(pred==255,19,pred) # pred가  255값, 즉 ingored value라면 19로 처리 아니면 pred 
        labels = np.where(labels==255, 19, labels)
        # 먼자 한줄로 펴서 라벨컬러랑 매칭시킴 그리고 다시 사이즈 복원
        #pred = np.reshape(pred,(-1,))
        
        index1 = np.reshape(pred, (-1,))
        index2 = np.reshape(labels, (-1,))

        one_hot1 = np.eye(num_classes)[index1]
        one_hot2 = np.eye(num_classes)[index2]
        
        pred_img = np.reshape(np.matmul(one_hot1, self.label_colors), (h, w, 3))
        gt_img= np.reshape(np.matmul(one_hot2, self.label_colors), (h, w, 3))
                              
        pred_show_img = (0.7*images+0.3*pred_img).astype('uint8')
        gt_show_img = (0.7*images+0.3*gt_img).astype('uint8')
                              
        # 2개의 영상을 붙여서 출력할것!
        pred_show_img=cv2.cvtColor(pred_show_img, cv2.COLOR_RGB2BGR)  #BGR로 붙여서 합침
        gt_show_img=cv2.cvtColor(gt_show_img, cv2.COLOR_RGB2BGR)

        save_img = np.concatenate([pred_show_img,gt_show_img], axis=1) # numpy에서는 concatenate full name 사용함
        dir_name = "./res_dir{}".format(epoch)
        img_name = os.path.join(dir_name, "Out_{:d}.png".format(e_step))
        #if not os.path.exists(self.res_dir):
            #os.mkdir(self.res_dir)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
           
        cv2.imwrite(img_name, save_img)
        
        return print('stored!',e_step)
    
    
# decay 추가되어야함
class LRController(object):
    
    def __init__(self, cfg):
        
        self.cfg = cfg
                
        self.decay_epoch = cfg.ld_epoch
        self.decay_width = cfg.num_epoch - self.decay_epoch
        self.learning_rate = cfg.learning_rate
            
    def get_lr(self, epoch):
                
        if epoch < self.decay_epoch:
            lr =  self.learning_rate
        else:
            lr = 0.5 * self.learning_rate * (1. - float(epoch - self.decay_epoch)/float(self.decay_width))
            print('Linear Decay - Learning Rate is {:1.8f}'.format(lr))
            
        return lr