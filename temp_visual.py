import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import matplotlib as mpl
plt.rcParams.update({'figure.dpi':150})


dict_stage = {'normal':0,'early':1,'intermediate':2,'advanced':3}
dict_eye = {'OD':0, 'OS':1}


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


def get_label_from_file(label_file):
    label = []
    for _, row in pd.read_excel(label_file).iterrows():
        label.append(row[1:].values.tolist())
    return label

def get_info_from_file(data_info):
    info = []
    for _, row in pd.read_excel(data_info).iterrows():
        temp_info_gender = row[1]
        temp_info_age = int(row[2])
        temp_info_eye = dict_eye[row[3]]
        temp_info_st =  dict_stage[row[4]]
        info.append([temp_info_gender, temp_info_age, temp_info_eye, temp_info_st])
    return info



def show_SE():
    # './data/STAGE_training/STAGE_training/training_GT/task1_GT_training.xlsx',
    # './data/STAGE_training/STAGE_training/training_GT/task2_GT_training.xlsx',
    # './data/STAGE_training/STAGE_training/training_GT/task3_GT_training.xlsx',
    
    data_info='./data/STAGE_training/STAGE_training/data_info_training.xlsx'
    infos = get_info_from_file(data_info)
    label_dir = './data/STAGE_training/STAGE_training/training_GT/task2_GT_training.xlsx'
    labels = get_label_from_file(label_dir)

    se_id = 0
    save_dir = './temp_vis_log/'
    os.makedirs(save_dir, exist_ok=True)
    for label, info in zip(labels, infos):
        fig = plt.figure()
        temp_map = pts2SEmap(label, info[2])
        
        plt.imshow(temp_map)
        plt.colorbar()
        se_id += 1

        plt.savefig(save_dir + 'SE_' + str(se_id)+'.png')

        
        # plt.show()
    
    pass


def show_PD():
    # './data/STAGE_training/STAGE_training/training_GT/task1_GT_training.xlsx',
    # './data/STAGE_training/STAGE_training/training_GT/task2_GT_training.xlsx',
    # './data/STAGE_training/STAGE_training/training_GT/task3_GT_training.xlsx',
    
    data_info='./data/STAGE_training/STAGE_training/data_info_training.xlsx'
    infos = get_info_from_file(data_info)
    label_dir = './data/STAGE_training/STAGE_training/training_GT/task3_GT_training.xlsx'
    labels = get_label_from_file(label_dir)

    se_id = 0
    save_dir = './temp_vis_log/'
    os.makedirs(save_dir, exist_ok=True)
    for label, info in zip(labels, infos):
        fig = plt.figure()
        temp_map = pts2SEmap(label, info[2])
        
        plt.imshow(temp_map)
        norm = mpl.colors.Normalize(vmin=4, vmax=10)
        plt.colorbar(mpl.cm.ScalarMappable(norm=norm))
        se_id += 1

        plt.savefig(save_dir + 'PD_' + str(se_id)+'.png')

        
        # plt.show()
    
    pass



if __name__ == '__main__':
    pass
    # show_SE()
    show_PD()
