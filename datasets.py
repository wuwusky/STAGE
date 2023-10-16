import os
import cv2
import glob
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random
import scipy.ndimage as sn
import p_tqdm

dict_stage = {'normal':0,'early':1,'intermediate':2,'advanced':3}
dict_eye = {'OD':0, 'OS':1}


# def load_img(img_dir):
#     src = cv2.imread(img_dir, 0)
#     src_roi = src[:700,:]
#     dst_roi = cv2.resize(src_roi, (256, 256))
#     return dst_roi

def int2id(id_int):
    id_str = '0000'
    id_int = str(int(id_int))
    id_str = id_str[:-len(id_int)] + id_int
    return id_str

def pts2SEmap(pts, eye):
    if int(eye) == 0: # OD
        se_map = np.zeros(shape=(9,9))
        ## line 1: 1-4
        se_map[0,3],se_map[0,4],se_map[0,5],se_map[0,6]=pts[0],pts[1],pts[2],pts[3]
        ## line 2: 5-10
        se_map[1,2],se_map[1,3],se_map[1,4],se_map[1,5],se_map[1,6],se_map[1,7]=\
        pts[4],pts[5],pts[6],pts[7],pts[8],pts[9]
        ## line 3, 11-18
        se_map[2,1],se_map[2,2],se_map[2,3],se_map[2,4],se_map[2,5],se_map[2,6],se_map[2,7],se_map[2,8]=\
        pts[10],pts[11],pts[12],pts[13],pts[14],pts[15],pts[16],pts[17]
        ## line 4, 19-26
        se_map[3,0],se_map[3,1],se_map[3,2],se_map[3,3],se_map[3,4],se_map[3,5],se_map[3,6],se_map[3,8]=\
        pts[18],pts[19],pts[20],pts[21],pts[22],pts[23],pts[24],pts[25]        
        ## line 5, 27-34
        se_map[4,0],se_map[4,1],se_map[4,2],se_map[4,3],se_map[4,4],se_map[4,5],se_map[4,6],se_map[4,8]=\
        pts[26],pts[27],pts[28],pts[29],pts[30],pts[31],pts[32],pts[33]
        ## line 6, 35-42
        se_map[5,1],se_map[5,2],se_map[5,3],se_map[5,4],se_map[5,5],se_map[5,6],se_map[5,7],se_map[5,8]=\
        pts[34],pts[35],pts[36],pts[37],pts[38],pts[39],pts[40],pts[41]
        ## line 7, 43-48
        se_map[6,2],se_map[6,3],se_map[6,4],se_map[6,5],se_map[6,6],se_map[6,7]=\
        pts[42],pts[43],pts[44],pts[45],pts[46],pts[47]
        ## line 8, 49-52
        se_map[7,3],se_map[7,4],se_map[7,5],se_map[7,6]=pts[48],pts[49],pts[50],pts[51]
    if int(eye) == 1: # OS
        se_map = np.zeros(shape=(9,9))
        ##line1
        se_map[0,2],se_map[0,3],se_map[0,4],se_map[0,5]=pts[3],pts[2],pts[1],pts[0]
        ##line2
        se_map[1,1],se_map[1,2],se_map[1,3],se_map[1,4],se_map[1,5],se_map[1,6]=pts[9],pts[8],pts[7],pts[6],pts[5],pts[4]
        ##line3
        se_map[2,0],se_map[2,1],se_map[2,2],se_map[2,3],se_map[2,4],se_map[2,5],se_map[2,6],se_map[2,7]=\
        pts[17],pts[16],pts[15],pts[14],pts[13],pts[12],pts[11],pts[10]
        ##line4
        se_map[3,0],se_map[3,2],se_map[3,3],se_map[3,4],se_map[3,5],se_map[3,6],se_map[3,7],se_map[3,8]=\
        pts[25],pts[24],pts[23],pts[22],pts[21],pts[20],pts[19],pts[18]
        ##line5
        se_map[4,0],se_map[4,2],se_map[4,3],se_map[4,4],se_map[4,5],se_map[4,6],se_map[4,7],se_map[4,8]=\
        pts[33],pts[32],pts[31],pts[30],pts[29],pts[28],pts[27],pts[26]
        ##line6
        se_map[5,0],se_map[5,1],se_map[5,2],se_map[5,3],se_map[5,4],se_map[5,5],se_map[5,6],se_map[5,7]=\
        pts[41],pts[40],pts[39],pts[38],pts[37],pts[36],pts[35],pts[34]
        ##line7
        se_map[6,1],se_map[6,2],se_map[6,3],se_map[6,4],se_map[6,5],se_map[6,6]=\
        pts[47],pts[46],pts[45],pts[44],pts[43],pts[42]
        ##line8
        se_map[7,2],se_map[7,3],se_map[7,4],se_map[7,5]=pts[51],pts[50],pts[49],pts[48]

    return se_map

def SEmap2pts(se_map, eye):
    # se_map = se_map.tolist()
    if int(eye) == 0: ## OD
        pts = []
        
        ##line1
        ## line 1: 1-4
        pts+=[se_map[0,3],se_map[0,4],se_map[0,5],se_map[0,6]]
        ## line 2, 5-10
        pts+=[se_map[1,2],se_map[1,3],se_map[1,4],se_map[1,5],se_map[1,6],se_map[1,7]]
        ## line 3, 11-18
        pts+=[se_map[2,1],se_map[2,2],se_map[2,3],se_map[2,4],se_map[2,5],se_map[2,6],se_map[2,7],se_map[2,8]]
        ## line 4, 19-26
        pts+=[se_map[3,0],se_map[3,1],se_map[3,2],se_map[3,3],se_map[3,4],se_map[3,5],se_map[3,6],se_map[3,8]]
        ## line 5, 27-34
        pts+=[se_map[4,0],se_map[4,1],se_map[4,2],se_map[4,3],se_map[4,4],se_map[4,5],se_map[4,6],se_map[4,8]]
        ## line 6, 35-42
        pts+=[se_map[5,1],se_map[5,2],se_map[5,3],se_map[5,4],se_map[5,5],se_map[5,6],se_map[5,7],se_map[5,8]]
        ## line 7, 43-48
        pts+=[se_map[6,2],se_map[6,3],se_map[6,4],se_map[6,5],se_map[6,6],se_map[6,7]]
        ## line 8, 49-52
        pts+=[se_map[7,3],se_map[7,4],se_map[7,5],se_map[7,6]]
    if int(eye) == 1: ## OS
        pts = []
        ##line1
        pts+=[se_map[0,5],se_map[0,4],se_map[0,3],se_map[0,2]]
        ##line2
        pts+=[se_map[1,6],se_map[1,5],se_map[1,4],se_map[1,3],se_map[1,2],se_map[1,1]]
        ##line3
        pts+=[se_map[2,7],se_map[2,6],se_map[2,5],se_map[2,4],se_map[2,3],se_map[2,2],se_map[2,1],se_map[2,0]]
        ##line4
        pts+=[se_map[3,8],se_map[3,7],se_map[3,6],se_map[3,5],se_map[3,4],se_map[3,3],se_map[3,2],se_map[3,0]]
        ##line5
        pts+=[se_map[4,8],se_map[4,7],se_map[4,6],se_map[4,5],se_map[4,4],se_map[4,3],se_map[4,2],se_map[4,0]]
        ##line6
        pts+=[se_map[5,7],se_map[5,6],se_map[5,5],se_map[5,4],se_map[5,3],se_map[5,2],se_map[5,1],se_map[5,0]]
        ##line7
        pts+=[se_map[6,6],se_map[6,5],se_map[6,4],se_map[6,3],se_map[6,2],se_map[6,1]]
        ##line8
        pts+=[se_map[7,5],se_map[7,4],se_map[7,3],se_map[7,2]]
    
    return pts


def random_crop(img_oct):
    crop_size = 500
    x = random.randint(0, 510-crop_size)
    y = random.randint(0, 510-crop_size)
    z = random.randint(0, 25)

    img_oct_ori = img_oct[z:z+230, y:y+crop_size, x:x+crop_size]
    return img_oct_ori

def center_crop(img_oct):
    crop_size = 500
    x = (510-crop_size)//2
    y = (510-crop_size)//2

    img_oct_ori = img_oct[12:242, y:y+crop_size, x:x+crop_size]
    return img_oct_ori

def random_shift(img_oct, max_shift=10):
    # d, h, w = img_oct.shape
    r_z = np.random.randint(-max_shift, max_shift+1)
    r_y = np.random.randint(-max_shift, max_shift+1)
    r_x = np.random.randint(-max_shift, max_shift+1)

    img_oct_shift = np.zeros_like(img_oct)
    img_oct_shift = sn.shift(img_oct, (r_z,r_y,r_x), mode='constant', cval=0)
    return img_oct_shift

def random_flip(img_oct):
    flag_od2os = []
    if random.random()<0.5:
        img_oct = np.flip(img_oct, axis=0)
        flag_od2os.append(0)
    if random.random()<0.5:
        img_oct = np.flip(img_oct, axis=1)
        flag_od2os.append(1)
    if random.random()<0.5:
        img_oct = np.flip(img_oct, axis=2)
        flag_od2os.append(2)
    
    return img_oct, flag_od2os

def random_eq(img_oct):
    d, h, w = img_oct.shape
    # alphas = np.random.uniform(0.8, 1.2, size=(d, h, w))
    # betas = np.random.uniform(-20, 20, size=(d, h, w))
    alphas = np.random.uniform(0.8, 1.2)
    betas = np.random.uniform(-10, 10)
    img_oct_eq = np.clip(alphas*img_oct+betas, 0, 255)
    return img_oct_eq

def random_noise(img_oct):
    noise = np.random.normal(0, 2, size=img_oct.shape)
    img_oct_noise = img_oct+noise
    img_oct_noise = np.clip(img_oct_noise, 0, 255)
    return img_oct_noise


class dataset_sub1(Dataset):
    def __init__(self,
                 dict_imgs,
                 label_file='',
                 data_info='',
                 mode='train',
                 data_scale=1
                 ):

        self.img_nps = dict_imgs
        self.mode = mode.lower()
        self.idx = list(dict_imgs.keys())

        if self.mode != 'test':
            label = {}
            for _, row in pd.read_excel(label_file).iterrows():
                label[int2id(row['ID'])]=row[1]
            info = {}
            for _, row in pd.read_excel(data_info).iterrows():
                temp_info_gender = row[1]
                temp_info_age = int(row[2])
                temp_info_eye = dict_eye[row[3]]
                temp_info_st =  dict_stage[row[4]]
                info[int2id(row['ID'])]=[temp_info_gender, temp_info_age, temp_info_eye, temp_info_st]
            self.label = label
            self.info = info
        else:
            info = {}
            for _, row in pd.read_excel(data_info).iterrows():
                temp_info_gender = row[1]
                temp_info_age = int(row[2])
                temp_info_eye = dict_eye[row[3]]
                temp_info_st =  dict_stage[row[4]]
                info[int2id(row['ID'])]=[temp_info_gender, temp_info_age, temp_info_eye, temp_info_st]
            self.info = info
        # print(self.idx[:10])

        if mode == 'train':
            list_labels = []
            for i, k in enumerate(dict_imgs.keys()):
                list_labels.append([i, k, label[k]])
            
            range_list = [(-35,-30), (-30, -25), (-25,20), (-20,-15), (-15, -10), (-10,-5), (-5, 5)]
            max_num = 0
            for range in range_list:
                values = []
                r_start, r_end = range
                for x in list_labels:
                    if r_start<=x[-1]<=r_end:
                        values.append(x)
                if len(values)>=max_num:
                    max_num=len(values)
            
            list_data_resample = []
            for range in range_list:
                r_start, r_end = range
                values = []
                temp_indexs = []
                for x in list_labels:
                    if r_start<=x[-1]<=r_end:
                        values.append(x[1:])
                        temp_indexs.append(x[0])
                
                if len(temp_indexs)<1:
                    continue
                # ind = np.arange(len(values))
                sub_ind = np.random.choice(temp_indexs, max_num, replace=True)
                for ind in sub_ind:
                    list_data_resample.append(list_labels[ind][1:])


            self.index = list_data_resample
        else:
            self.index = self.idx


    def __getitem__(self, idx):
        
        if self.mode != 'test':
            if self.mode == 'train':
                real_index = self.index[idx][0]
            else:
                real_index = self.index[idx]
            label = self.label[real_index]
            label = torch.from_numpy(np.array([label])).float()
        else:
            real_index = self.index[idx]
        info = self.info[real_index]
        flag_od2os = 0
        
        oct_img = self.img_nps[real_index].astype(np.float32)
        if self.mode == 'train':
            # if random.random()<0.5:
            #     oct_img = random_noise(oct_img.copy())
            if random.random()<0.5:
                oct_img, flag_od2os = random_flip(oct_img.copy())
            if random.random()<0.5:
                oct_img = random_eq(oct_img.copy())
            
            # if random.random()<0.5:
            #     oct_img = random_shift(oct_img.copy())

            oct_img = center_crop(oct_img.copy())
        else:
            oct_img = center_crop(oct_img.copy())

        oct_img = torch.from_numpy(np.ascontiguousarray(oct_img)).float()
        

        data = {}
        data['img'] = oct_img
        data['img_id'] = real_index
        if self.mode != 'test':
            data['MD'] = label + random.uniform(-0.5, 0.5)
        data['stage'] = torch.from_numpy(np.array([info[3]])).float()
        data['age'] = torch.from_numpy(np.array([info[1]])).float()
        if flag_od2os==1:
            data['eye'] = 1-torch.from_numpy(np.array([info[2]])).float()
        else:
            data['eye'] = torch.from_numpy(np.array([info[2]])).float()
        return data

    def __len__(self):
        return len(self.index)


class dataset_sub2(Dataset):
    def __init__(self, 
                 dict_imgs,
                 label_file='',
                 data_info='',
                 mode='train',
                 data_scale=1
                 ):
        self.img_nps = dict_imgs
        self.mode = mode.lower()
        self.idx = list(dict_imgs.keys())*data_scale

        if self.mode == 'train' or self.mode == 'valid':
            label = {}
            for _, row in pd.read_excel(label_file).iterrows():
                label[int2id(row['ID'])]=row[1:].values.tolist()

            info = {}
            for _, row in pd.read_excel(data_info).iterrows():
                temp_info_gender = row[1]
                temp_info_age = int(row[2])
                temp_info_eye = dict_eye[row[3]]
                temp_info_st =  dict_stage[row[4]]
                info[int2id(row['ID'])]=[temp_info_gender, temp_info_age, temp_info_eye, temp_info_st]
            self.label = label
            self.info = info


        elif self.mode == "test":
            info = {}
            for _, row in pd.read_excel(data_info).iterrows():
                temp_info_gender = row[1]
                temp_info_age = int(row[2])
                temp_info_eye = dict_eye[row[3]]
                temp_info_st =  dict_stage[row[4]]
                info[int2id(row['ID'])]=[temp_info_gender, temp_info_age, temp_info_eye, temp_info_st]
            self.info = info

        
    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        real_index = self.idx[idx]
        if self.mode == 'train' or self.mode=='valid':
            label = self.label[real_index]
            
        info = self.info[real_index]
        flag_od2os = []
        
        oct_img = self.img_nps[real_index]
        if self.mode == 'train':
            # if random.random()<0.5:
            #     oct_img = random_noise(oct_img.copy())
            if random.random()<0.5:
                oct_img, flag_od2os = random_flip(oct_img.copy())
            # if random.random()<0.5:
            #     oct_img = random_eq(oct_img.copy())
            
            # if random.random()<0.5:
            #     oct_img = random_shift(oct_img.copy())

            oct_img = center_crop(oct_img.copy())
        else:
            oct_img = center_crop(oct_img.copy())

        oct_img = torch.from_numpy(np.ascontiguousarray(oct_img)).float()

        data = {}
        # z, y, x

        data['img'] = oct_img
        data['img_id'] = real_index
        if self.mode != 'test':
            label = pts2SEmap(label, info[2])
            for temp_index in flag_od2os:
                if temp_index==0:
                    label = np.flip(label.copy(), axis=0)
                if temp_index==2:
                    label = np.flip(label.copy(), axis=1)
            label = torch.from_numpy(label.copy()).float()
            data['SE'] = label
        data['stage'] = torch.from_numpy(np.array([info[3]])).float()
        data['age'] = torch.from_numpy(np.array([info[1]])).float()

        data['eye'] = torch.from_numpy(np.array([info[2]])).float()

        return data


class dataset_sub3(Dataset):
    def __init__(self, 
                 dict_imgs,
                 label_file='',
                 data_info='',
                 mode='train',
                 data_scale=1
                 ):
        self.img_nps = dict_imgs
        self.mode = mode.lower()
        self.idx = list(dict_imgs.keys())*data_scale

        if self.mode == 'train' or self.mode == 'valid':
            label = {}
            for _, row in pd.read_excel(label_file).iterrows():
                label[int2id(row['ID'])]=row[1:].values.tolist()

            info = {}
            for _, row in pd.read_excel(data_info).iterrows():
                temp_info_gender = row[1]
                temp_info_age = int(row[2])
                temp_info_eye = dict_eye[row[3]]
                temp_info_st =  dict_stage[row[4]]
                info[int2id(row['ID'])]=[temp_info_gender, temp_info_age, temp_info_eye, temp_info_st]
            self.label = label
            self.info = info


        elif self.mode == "test":
            info = {}
            for _, row in pd.read_excel(data_info).iterrows():
                temp_info_gender = row[1]
                temp_info_age = int(row[2])
                temp_info_eye = dict_eye[row[3]]
                temp_info_st =  dict_stage[row[4]]
                info[int2id(row['ID'])]=[temp_info_gender, temp_info_age, temp_info_eye, temp_info_st]
            self.info = info
        
        
    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        real_index = self.idx[idx]
        if self.mode == 'train' or self.mode=='valid':
            label = self.label[real_index]
            label = torch.from_numpy(np.array([label])).float()[0]
        info = self.info[real_index]
        flag_od2os = 0
        
        oct_img = self.img_nps[real_index]
        if self.mode == 'train':
            
            if random.random()<0.5:
                oct_img, flag_od2os = random_flip(oct_img.copy())
            if random.random()<0.5:
                oct_img = random_eq(oct_img.copy())


            oct_img = random_crop(oct_img.copy())
        else:
            oct_img = center_crop(oct_img)
        oct_img = torch.from_numpy(np.ascontiguousarray(oct_img)).float()

        data = {}
        data['img'] = oct_img
        data['img_id'] = real_index
        if self.mode != 'test':
            if flag_od2os==1:
                label = torch.flip(label, dims=[-1])
            data['PD_train'] = label + random.uniform(-0.1, 0.1)
            data['PD'] = label
        data['stage'] = torch.from_numpy(np.array([info[3]])).float()

        data['age'] = torch.from_numpy(np.array([info[1]])).float()
        if flag_od2os==1:
            data['eye'] = 1-torch.from_numpy(np.array([info[2]])).float()
        else:
            data['eye'] = torch.from_numpy(np.array([info[2]])).float()
        return data


def get_label_from_file(label_file):
    label = {}
    for _, row in pd.read_excel(label_file).iterrows():
        label[int2id(row['ID'])]=row[1:].values.tolist()
    return label

def get_info_from_file(data_info):
    info = {}
    for _, row in pd.read_excel(data_info).iterrows():
        temp_info_gender = row[1]
        temp_info_age = int(row[2])
        temp_info_eye = dict_eye[row[3]]
        temp_info_st =  dict_stage[row[4]]
        info[int2id(row['ID'])]=[temp_info_gender, temp_info_age, temp_info_eye, temp_info_st]
    return info

class dataset_mt(Dataset):
    def __init__(self, 
                 dict_imgs,
                 label_files=None,
                 data_info=None,
                 mode='train',
                 data_scale=1
                 ):
        self.img_nps = dict_imgs
        self.mode = mode.lower()
        self.idx = list(dict_imgs.keys())*data_scale

        if self.mode != 'test':
            label_file_MD, label_file_SE, label_file_PD = label_files
            self.label_MD = get_label_from_file(label_file_MD)
            self.label_SE = get_label_from_file(label_file_SE)
            self.label_PD = get_label_from_file(label_file_PD)
            

        self.info = get_info_from_file(data_info)

        
    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        real_index = self.idx[idx]
        if self.mode != 'test':
            label_MD = self.label_MD[real_index]
            label_SE = self.label_SE[real_index]
            label_PD = self.label_PD[real_index]
            
        info = self.info[real_index]
        flag_od2os = []
        oct_img = self.img_nps[real_index]
        if self.mode == 'train':
            # if random.random()<0.5:
            #     oct_img = random_noise(oct_img.copy())
            if random.random()<0.5:
                oct_img, flag_od2os = random_flip(oct_img.copy())
            if random.random()<0.5:
                oct_img = random_eq(oct_img.copy())
            pass
            oct_img = center_crop(oct_img.copy())
        else:
            oct_img = center_crop(oct_img.copy())

        
        oct_img = torch.from_numpy(np.ascontiguousarray(oct_img)).float()# z, y, x
        
        oct_img = oct_img.permute(1,0,2)# y, z, x
        

        data = {}
        

        data['img'] = oct_img
        data['img_id'] = real_index
        if self.mode != 'test':
            ## MD
            label_MD = torch.from_numpy(np.array([label_MD])).float()
            data['MD'] = label_MD + random.uniform(-0.3, 0.3)
            
            ## SE
            label_SE = pts2SEmap(label_SE, info[2])
            for temp_index in flag_od2os:
                if temp_index==0:
                    label_SE = np.flip(label_SE.copy(), axis=0)
                if temp_index==2:
                    label_SE = np.flip(label_SE.copy(), axis=1)
            label_SE = torch.from_numpy(label_SE.copy()).float()
            data['SE'] = label_SE

            ## PD
            label_PD = pts2SEmap(label_PD, info[2])
            for temp_index in flag_od2os:
                if temp_index==0:
                    label_PD = np.flip(label_PD.copy(), axis=0)
                if temp_index==2:
                    label_PD = np.flip(label_PD.copy(), axis=1)
            label_PD = torch.from_numpy(label_PD.copy()).float()
            data['PD'] = label_PD

        
        data['age'] = torch.from_numpy(np.array([info[1]])).float()
        data['eye'] = torch.from_numpy(np.array([info[2]])).float()
        data['stage'] = torch.from_numpy(np.array([info[3]])).float()

        return data



def load_img(img_dir, size=512):
    src = cv2.imread(img_dir, 0)
    src_roi = src[:700,:]
    dst_roi = cv2.resize(src_roi, (size, size))
    return dst_roi

def load_test_data(test_root, size=512):
    filelists = os.listdir(test_root)
    train_filelists = []
    list_imgIDs = []
    for f in filelists:
        if 'DS' in f:
            continue
        else:
            train_filelists.append(test_root + '/' + f)
            list_imgIDs.append(f)

    def get_img_np(f_dir):
        list_fs = os.listdir(f_dir)
        list_fnames = []
        for fs in list_fs:
            if 'DS' in fs:
                continue
            else:
                list_fnames.append(fs)
        oct_series_list = sorted(list_fnames, key=lambda x: int(x.split("_")[0]))

        list_series = []

        for k, p in enumerate(oct_series_list):
            list_series.append(load_img(f_dir + '/' + p, size))
        
        oct_img = np.stack(list_series, axis=0)
        return oct_img
    
    list_img_nps = []

    result = p_tqdm.p_umap(get_img_np, train_filelists, ncols=100)
    for res in result:
        list_img_nps.append(res)

    data_dict = {}
    for id, data in zip(list_imgIDs, list_img_nps):
        data_dict[id] = data
    return data_dict

class dataset_mt_scratch(Dataset):
    def __init__(self, 
                 test_root,
                 label_files=None,
                 data_info=None,
                 mode='train',
                 data_scale=1
                 ):
        
        dict_imgs = load_test_data(test_root)
        self.img_nps = dict_imgs
        self.mode = mode.lower()
        self.idx = list(dict_imgs.keys())

        if self.mode != 'test':
            label_file_MD, label_file_SE, label_file_PD = label_files
            self.label_MD = get_label_from_file(label_file_MD)
            self.label_SE = get_label_from_file(label_file_SE)
            self.label_PD = get_label_from_file(label_file_PD)
            

        self.info = get_info_from_file(data_info)

        
    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        real_index = self.idx[idx]
        if self.mode != 'test':
            label_MD = self.label_MD[real_index]
            label_SE = self.label_SE[real_index]
            label_PD = self.label_PD[real_index]
            
        info = self.info[real_index]
        flag_od2os = []
        oct_img = self.img_nps[real_index]
        if self.mode == 'train':
            # if random.random()<0.5:
            #     oct_img = random_noise(oct_img.copy())
            if random.random()<0.5:
                oct_img, flag_od2os = random_flip(oct_img.copy())
            if random.random()<0.5:
                oct_img = random_eq(oct_img.copy())
            pass
            oct_img = center_crop(oct_img.copy())
        else:
            oct_img = center_crop(oct_img.copy())

        
        oct_img = torch.from_numpy(np.ascontiguousarray(oct_img)).float()# z, y, x
        
        oct_img = oct_img.permute(1,0,2)# y, z, x
        

        data = {}
        

        data['img'] = oct_img
        data['img_id'] = real_index
        if self.mode != 'test':
            ## MD
            label_MD = torch.from_numpy(np.array([label_MD])).float()
            data['MD'] = label_MD + random.uniform(-0.3, 0.3)
            
            ## SE
            label_SE = pts2SEmap(label_SE, info[2])
            for temp_index in flag_od2os:
                if temp_index==0:
                    label_SE = np.flip(label_SE.copy(), axis=0)
                if temp_index==2:
                    label_SE = np.flip(label_SE.copy(), axis=1)
            label_SE = torch.from_numpy(label_SE.copy()).float()
            data['SE'] = label_SE

            ## PD
            label_PD = pts2SEmap(label_PD, info[2])
            for temp_index in flag_od2os:
                if temp_index==0:
                    label_PD = np.flip(label_PD.copy(), axis=0)
                if temp_index==2:
                    label_PD = np.flip(label_PD.copy(), axis=1)
            label_PD = torch.from_numpy(label_PD.copy()).float()
            data['PD'] = label_PD

        
        data['age'] = torch.from_numpy(np.array([info[1]])).float()
        data['eye'] = torch.from_numpy(np.array([info[2]])).float()
        data['stage'] = torch.from_numpy(np.array([info[3]])).float()

        return data

