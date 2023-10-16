from datasets import dataset_sub1, dataset_sub2, dataset_sub3, dataset_mt, dataset_mt_scratch
from datasets import pts2SEmap, SEmap2pts
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from models import resnet18, resnet50, resnet34
from tqdm import tqdm
import cv2
import p_tqdm
from sklearn import metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


trainset_root = "./data/STAGE_training/STAGE_training/training_images"
# test_root = "./data/STAGE_validation/STAGE_validation/validation_images"
test_root = 'data/STAGE_FINAL_IMGS/STAGE_FINAL_IMGS/final_imgs/'


class SMAPELoss(nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, preds, labels):
        smap = 1 / len(preds) * torch.sum(2 * torch.abs(preds - labels) / (torch.abs(preds) + torch.abs(labels)+1e-6))
        # smap = torch.abs(0.9-smap)
        return smap

class R2Loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, preds, labels):
        r2 = 1 - torch.sum((preds - labels+1e-6) ** 2) / torch.sum((labels - torch.mean(labels)+1e-6) ** 2)
        r2 = 1-r2
        return r2

def Smape_(preds, labels):
    return 1 / len(preds) * np.sum(2*np.abs(preds-labels)/(np.abs(preds)+np.abs(labels)+1e-6))

def r2_score_func(y_true, y_pred):
    a = np.square(y_pred - y_true)
    b = np.sum(a)
    c = np.mean(y_true)
    d = np.square(y_true - c)
    e = np.sum(d)
    f = 1 - b/(e + 1e-12)
    return f

def R2_(preds, labels):
    return 1 - np.sum((preds - labels+1e-6) ** 2) / np.sum((labels - np.mean(labels)+1e-6) ** 2)

def Score(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    smape = Smape_(preds, labels)
    R2 = R2_(preds, labels)
    score = 0.5 * (1 / (smape + 0.1)) + 0.5 * (R2 * 10)
    return  score, smape, R2



def Score_s3(preds, labels):
    preds = np.array(preds).astype(np.uint8).reshape(-1)
    preds[preds<0]=0
    preds[preds>4]=4
    labels = np.array(labels).reshape(-1)
    
    f1 = metrics.f1_score(labels, preds, average='macro')
    return  f1

def load_img(img_dir, size=512):
    src = cv2.imread(img_dir, 0)
    src_roi = src[:700,:]
    dst_roi = cv2.resize(src_roi, (size, size))
    return dst_roi



def generate_dataset_train(size=512):
    filelists = os.listdir(trainset_root)
    train_filelists = []
    for f in filelists:
        if 'DS' in f:
            continue
        else:
            train_filelists.append(trainset_root + '/' + f)

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

    temp_save_dir = './dataset_' + str(size) +'.npy' 
    np.save(temp_save_dir, list_img_nps)
    print('generate finished~')


def generate_dataset_test(size=512):
    filelists = os.listdir(test_root)
    train_filelists = []
    for f in filelists:
        if 'DS' in f:
            continue
        else:
            train_filelists.append(test_root + '/' + f)

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

    temp_save_dir = './dataset_' + str(size) +'_test.npy' 
    np.save(temp_save_dir, list_img_nps)
    print('generate finished~')


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                try:
                    norm = torch.norm(param.grad)
                    if norm != 0 and not torch.isnan(norm):
                        r_at = epsilon * param.grad / norm
                        param.data.add_(r_at)
                except Exception as e:
                    continue

    def restore(self, emb_name='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, prediction, label):
        mask = torch.ones_like(label)
        mask[label == 0] = 5
        loss = torch.mean(mask * (prediction - label) ** 2)
        return loss

class PDMSELoss(nn.Module):
    def __init__(self):
        super(PDMSELoss, self).__init__()
    def forward(self, prediction, label):
        mask=torch.ones_like(label)
        mask[label>0]=10
        loss = torch.mean(mask*(prediction-label)**2)
        return loss

class PDl1Loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, prediction, label):
        mask=torch.ones_like(label)
        mask[label>0]=10
        loss = torch.mean(mask*torch.abs(prediction-label))
        return loss

def task1(only_eval=False):
    # ratio_stage_md=5.0
    log_dir = './logs/'
    os.makedirs(log_dir, exist_ok=True)
    model = resnet18(dim_pred=1, in_ch=230).to(device)
    model_eval = resnet18(dim_pred=1, in_ch=230)

    if only_eval ==False:
        filelists = os.listdir(trainset_root)
        train_filelists = []
        for f in filelists:
            if 'DS' in f:
                continue
            else:
                train_filelists.append(f)
        
        list_data = np.load('./dataset_'+str(re_size)+'.npy')
        print('load dataset finished~')
        dict_imgs_train = {}
        dict_imgs_valid = {}

        for i in range(len(train_filelists)):
            f_id = train_filelists[i]
            oct = list_data[i]
            if (i+1)%(1/val_ratio)==0:
                dict_imgs_valid[f_id] = oct
            else:
                dict_imgs_train[f_id] = oct


        dataset_train = dataset_sub1(
                            dict_imgs=dict_imgs_train,
                            label_file='./data/STAGE_training/STAGE_training/training_GT/task1_GT_training.xlsx',
                            data_info='./data/STAGE_training/STAGE_training/data_info_training.xlsx',
                            mode='train',
                            data_scale=2,
                            )
        loader_train = DataLoader(dataset_train, batch_size, True, num_workers=num_workers, drop_last=True)

        dataset_valid = dataset_sub1(
                            dict_imgs=dict_imgs_valid,
                            label_file='./data/STAGE_training/STAGE_training/training_GT/task1_GT_training.xlsx',
                            data_info='./data/STAGE_training/STAGE_training/data_info_training.xlsx',
                            mode='valid',
                            )
        loader_valid = DataLoader(dataset_valid, batch_size, False, num_workers=num_workers, drop_last=False)

        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
        lr_sh = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[epoch_max*2//3], gamma=0.1)
        loss_fun = nn.SmoothL1Loss()
        loss_fun_sm = SMAPELoss()
        loss_fun_r2 = R2Loss()

        best_score = 0

        for epoch in range(epoch_max):
            model.train()
            list_gts = []
            list_preds = []
            with tqdm(loader_train, ncols=150) as tqdmDataLoader:
                for i, data in enumerate(tqdmDataLoader):

                    img = data['img'].to(device)
                    lbl = data['MD'].to(device)
                    st = data['stage'].to(device)
                    eye = data['eye'].to(device)
                    age = data['age'].to(device)

                    out = model.forward_md(img, st, eye, age)
                    # pred = -st*ratio_stage_md+out
                    pred = out
                    b = pred.shape[0]


                    loss_l1 = loss_fun(pred.view(b,-1), lbl.view(b,-1))
                    loss_sm = loss_fun_sm(pred.view(b,-1), lbl.view(b,-1))
                    loss_r2 = loss_fun_r2(pred.view(b,-1), lbl.view(b,-1))

                    loss = loss_l1 + loss_r2 + loss_sm
                    loss.backward()
                    optim.step()
                    optim.zero_grad()

                    tqdmDataLoader.set_postfix(ordered_dict={
                                        "Epoch":epoch+1,
                                        "L_l1": loss_l1.item(),
                                        "L_sm": loss_sm.item(),
                                        "L_r2": loss_r2.item(),
                                        "LR":optim.param_groups[0]['lr'],
                                    })
                    list_gts += lbl.cpu().view(-1).numpy().tolist()
                    list_preds += pred.detach().cpu().view(-1).numpy().tolist()

            lr_sh.step()
            train_score, train_smap, train_r2 = Score(list_preds, list_gts)
            # print(len(list_preds))
            # print(len(list_gts))
            plt.figure()
            seaborn.scatterplot(x=list_preds, y=list_gts)
            t_min, t_max = np.min(list_gts), np.max(list_gts)
            plt.xlim([t_min, t_max])
            plt.savefig(log_dir+'./MD_train.png')
            

            model.eval()
            list_gts = []
            list_preds = []
            for data in tqdm(loader_valid, ncols=50):
                img = data['img'].to(device)
                lbl = data['MD']
                st = data['stage'].to(device)
                eye = data['eye'].to(device)
                age = data['age'].to(device)

                    

                with torch.no_grad():
                    out = model.forward_md(img, st, eye, age)
                # pred = -st*ratio_stage_md+out
                pred = out
                
                list_gts += lbl.cpu().view(-1).numpy().tolist()
                list_preds += pred.detach().cpu().view(-1).numpy().tolist()
            temp_score, temp_smap, temp_r2 = Score(list_preds, list_gts)
            plt.figure()
            seaborn.scatterplot(x=list_preds, y=list_gts)
            t_min, t_max = np.min(list_gts), np.max(list_gts)
            plt.xlim([t_min, t_max])
            plt.savefig(log_dir+'./MD_valid.png')

            print('train Score:{:.4f}, smape:{:.4f}, r2:{:.4f}'.format(train_score, train_smap, train_r2))
            print('valid Score:{:.4f}, smape:{:.4f}, r2:{:.4f}'.format(temp_score, temp_smap, temp_r2))

            if temp_score >= best_score:
                best_score = temp_score
                torch.save(model.state_dict() ,'./model_t1_best.pb')
                print('Best Score:{:.4f}, smape:{:.4f}, r2:{:.4f}---------------------------------------------------------good job--------------------------------------------------------'.format(temp_score, temp_smap, temp_r2))
                plt.figure()
                seaborn.scatterplot(x=list_preds, y=list_gts)
                t_min, t_max = np.min(list_gts), np.max(list_gts)
                plt.xlim([t_min, t_max])
                plt.savefig(log_dir+'./MD_valid_best.png')
            torch.save(model.state_dict(), './model_t1.pb')
            plt.cla()
            plt.close('all')

    filelists = os.listdir(test_root)
    test_filelists = []
    for f in filelists:
        if 'DS' in f:
            continue
        else:
            test_filelists.append(f)
    
    list_data = np.load('./dataset_'+str(re_size)+'_test.npy')
    print('load dataset finished~')
    dict_imgs = {}

    for f_id, oct in zip(test_filelists, list_data):
        dict_imgs[f_id] = oct



    filelists = os.listdir(test_root)
    test_dataset = dataset_sub1(dict_imgs,
                                data_info='./data/STAGE_validation/STAGE_validation/data_info_validation.xlsx',
                                mode='test')
    loader_test = DataLoader(test_dataset, 1, False, num_workers=0, drop_last=False)

    
    model_eval.load_state_dict(torch.load('./model_t1_best.pb', map_location='cpu'), strict=True)
    model_eval = model_eval.to(device)
    model_eval.eval()

    list_results = []
    list_vs = []
    for data in tqdm(loader_test, ncols=150):
        img = data['img'].to(device)
        idx = data['img_id']
        st = data['stage'].to(device)
        eye = data['eye'].to(device)
        age = data['age'].to(device)
        

        with torch.no_grad():
            out = model.forward_md(img, st, eye, age)
        # pred = -st*ratio_stage_md+out
        pred = out
        pred = pred.detach().cpu().view(-1).numpy()
        temp_results = [int(idx[0])] + pred.tolist()

        list_results.append(temp_results)
        list_vs += pred.tolist()
    
    list_results.sort(key=lambda x:x[0])
    submission_result = pd.DataFrame(list_results, columns=['ID', 'pred_MD'])
    submission_result.to_csv("./MD_Results.csv", index=False)

    plt.figure()
    seaborn.histplot(list_vs)
    plt.savefig(log_dir+'./MD_test_hist.png')
    plt.cla()
    plt.close('all')


def task2(only_eval=False):
    log_dir = './logs/'
    os.makedirs(log_dir, exist_ok=True)
    model = resnet18(dim_pred=52).to(device)
    model_eval = resnet18(dim_pred=52)

    if only_eval==False:
        filelists = os.listdir(trainset_root)
        train_filelists = []
        for f in filelists:
            if 'DS' in f:
                continue
            else:
                train_filelists.append(f)
        
        list_data = np.load('./dataset_'+str(re_size)+'.npy')
        print('load dataset finished~')
        dict_imgs_train = {}
        dict_imgs_valid = {}

        for i in range(len(train_filelists)):
            f_id = train_filelists[i]
            oct = list_data[i]
            if (i+1)%(1/val_ratio)==0:
                dict_imgs_valid[f_id] = oct
            else:
                dict_imgs_train[f_id] = oct


        dataset_train = dataset_sub2(
                            dict_imgs=dict_imgs_train,
                            label_file='./data/STAGE_training/STAGE_training/training_GT/task2_GT_training.xlsx',
                            data_info='./data/STAGE_training/STAGE_training/data_info_training.xlsx',
                            mode='train',
                            data_scale=10,
                            )
        loader_train = DataLoader(dataset_train, batch_size, True, num_workers=num_workers, drop_last=True, pin_memory=False)

        dataset_valid = dataset_sub2(
                            dict_imgs=dict_imgs_valid,
                            label_file='./data/STAGE_training/STAGE_training/training_GT/task2_GT_training.xlsx',
                            data_info='./data/STAGE_training/STAGE_training/data_info_training.xlsx',
                            mode='train',
                            )
        loader_valid = DataLoader(dataset_valid, batch_size, False, num_workers=8, drop_last=False)

        
        # model.load_state_dict(torch.load('./model_t2.pb', map_location='cpu'), strict=True)
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
        lr_sh = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[epoch_max*2//3], gamma=0.1)
        loss_fun = nn.SmoothL1Loss()
        loss_fun_sm = SMAPELoss()
        loss_fun_r2 = R2Loss()

        best_score = 0

        for epoch in range(epoch_max):
            model.train()
            list_gts = []
            list_preds = []
            with tqdm(loader_train, ncols=150) as tqdmDataLoader:
                for i, data in enumerate(tqdmDataLoader):

                    img = data['img'].to(device)
                    lbl = data['SE'].to(device)
                    st = data['stage'].to(device)
                    eye = data['eye'].to(device)
                    age = data['age'].to(device)

                    out = model.forward_se(img, st, eye, age)
                    b = out.shape[0]

                    loss_l1 = loss_fun(out.view(b,-1), lbl.view(b,-1))
                    loss_sm = loss_fun_sm(out.view(b,-1), lbl.view(b,-1))
                    loss_r2 = loss_fun_r2(out.view(b,-1), lbl.view(b,-1))

                    loss = loss_l1 + loss_sm*0.01 + loss_r2
                    # loss = loss_sm + loss_r2

                    loss.backward()
                    optim.step()
                    optim.zero_grad()

                    tqdmDataLoader.set_postfix(ordered_dict={
                                        "Epoch":epoch+1,
                                        "L_l1": loss_l1.item(),
                                        "L_sm": loss_sm.item(),
                                        "L_r2": loss_r2.item(),
                                        "LR":optim.param_groups[0]['lr'],
                                    })
                    list_gts += lbl.cpu().view(-1).numpy().tolist()
                    list_preds += out.detach().cpu().view(-1).numpy().tolist()

            lr_sh.step()
            train_score, train_smap, train_r2 = Score(list_preds, list_gts)
            plt.figure()
            seaborn.scatterplot(x=list_preds, y=list_gts)
            t_min, t_max = np.min(list_gts), np.max(list_gts)
            plt.xlim([t_min-5, t_max+5])
            plt.savefig(log_dir+'./SE_train.png')

            model.eval()
            list_gts = []
            list_preds = []
            for data in tqdm(loader_valid, ncols=150):
                img = data['img'].to(device)
                lbl = data['SE']
                st = data['stage'].to(device)
                eye = data['eye'].to(device)
                age = data['age'].to(device)

                    

                with torch.no_grad():
                    out = model.forward_se(img, st, eye, age)

                list_gts += lbl.view(-1).numpy().tolist()
                list_preds += out.detach().cpu().view(-1).numpy().tolist()
            temp_score, temp_smap, temp_r2 = Score(list_preds, list_gts)
            plt.figure()
            seaborn.scatterplot(x=list_preds, y=list_gts)
            t_min, t_max = np.min(list_gts), np.max(list_gts)
            plt.xlim([t_min-5, t_max+5])
            plt.savefig(log_dir+'./SE_valid.png')

            print('train Score:{:.4f}, smape:{:.4f}, r2:{:.4f}'.format(train_score, train_smap, train_r2))
            print('valid Score:{:.4f}, smape:{:.4f}, r2:{:.4f}'.format(temp_score, temp_smap, temp_r2))

            if temp_score >= best_score:
                best_score = temp_score
                torch.save(model.state_dict() ,'./model_t2_best.pb')
                print('Best Score:{:.4f}, smape:{:.4f}, r2:{:.4f}---------------------------------------------------------good job--------------------------------------------------------'.format(temp_score, temp_smap, temp_r2))
                plt.figure()
                seaborn.scatterplot(x=list_preds, y=list_gts)
                t_min, t_max = np.min(list_gts), np.max(list_gts)
                plt.xlim([t_min-5, t_max+5])
                plt.savefig(log_dir+'./SE_valid_best.png')
            torch.save(model.state_dict(), './model_t2.pb')
            plt.cla()
            plt.close('all')

    
    
    filelists = os.listdir(test_root)
    test_filelists = []
    for f in filelists:
        if 'DS' in f:
            continue
        else:
            test_filelists.append(f)
    
    list_data = np.load('./dataset_'+str(re_size)+'_test.npy')
    print('load dataset finished~')
    dict_imgs = {}

    for f_id, oct in zip(test_filelists, list_data):
        dict_imgs[f_id] = oct



    filelists = os.listdir(test_root)
    test_dataset = dataset_sub2(dict_imgs,
                                data_info='./data/STAGE_validation/STAGE_validation/data_info_validation.xlsx',
                                mode='test')
    loader_test = DataLoader(test_dataset, 1, False, num_workers=0, drop_last=False)

    
    model_eval.load_state_dict(torch.load('./model_t2_best.pb', map_location='cpu'), strict=True)
    model_eval = model_eval.to(device)
    model_eval.eval()

    list_results = []
    for data in tqdm(loader_test, ncols=150):
        img = data['img'].to(device)
        idx = data['img_id']
        st = data['stage'].to(device)
        eye = data['eye'].to(device)
        age = data['age'].to(device)

        with torch.no_grad():
            out = model_eval.forward_se(img, st, eye, age)

        out = out[0][0].detach().cpu().numpy()
        out[out<0]=0
        # print(out.shape)
        out = SEmap2pts(out, eye)
        # print(len(out))
        temp_results = [int(idx[0])] + out
        # print(temp_results)
        list_results.append(temp_results)

    list_results.sort(key=lambda x:x[0])
    temp_cols = ['ID']
    for i in range(52):
        temp_cols.append('point'+str(i+1))
    submission_result = pd.DataFrame(list_results, columns=temp_cols)
    submission_result.to_csv("./Sensitivity_map_Results.csv", index=False)


def task3():
    log_dir = './logs/'
    os.makedirs(log_dir, exist_ok=True)
    model = resnet18(dim_pred=52, in_ch=230).to(device)
    model_eval = resnet18(dim_pred=52, in_ch=230)


    filelists = os.listdir(trainset_root)
    train_filelists = []
    for f in filelists:
        if 'DS' in f:
            continue
        else:
            train_filelists.append(f)
    
    list_data = np.load('./dataset_'+str(re_size)+'.npy')
    print('load dataset finished~')
    dict_imgs_train = {}
    dict_imgs_valid = {}

    for i in range(len(train_filelists)):
        f_id = train_filelists[i]
        oct = list_data[i]
        if (i+1)%(1/val_ratio)==0:
            dict_imgs_valid[f_id] = oct
        else:
            dict_imgs_train[f_id] = oct


    dataset_train = dataset_sub3(
                        dict_imgs=dict_imgs_train,
                        label_file='./data/STAGE_training/STAGE_training/training_GT/task3_GT_training.xlsx',
                        data_info='./data/STAGE_training/STAGE_training/data_info_training.xlsx',
                        mode='train',
                        data_scale=2,
                        )
    loader_train = DataLoader(dataset_train, batch_size, True, num_workers=num_workers, drop_last=True)

    dataset_valid = dataset_sub3(
                        dict_imgs=dict_imgs_valid,
                        label_file='./data/STAGE_training/STAGE_training/training_GT/task3_GT_training.xlsx',
                        data_info='./data/STAGE_training/STAGE_training/data_info_training.xlsx',
                        mode='train',
                        )
    loader_valid = DataLoader(dataset_valid, batch_size, False, num_workers=num_workers, drop_last=False)


    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    lr_sh = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[epoch_max*2//3], gamma=0.1)
    loss_fun = PDMSELoss()
    # loss_fun_sm = SMAPELoss()
    best_score = 0

    for epoch in range(epoch_max):
        model.train()
        list_gts = []
        list_preds = []
        with tqdm(loader_train, ncols=150) as tqdmDataLoader:
            for i, data in enumerate(tqdmDataLoader):

                img = data['img'].to(device)
                lbl = data['PD'].to(device)
                lbl_train = data['PD_train'].to(device)
                st = data['stage'].to(device)
                eye = data['eye'].to(device)
                age = data['age'].to(device)

                out = model.forward_pd(img, st, eye, age)

                loss_l1 = loss_fun(out, lbl_train)


                loss = loss_l1

                loss.backward()
                optim.step()
                optim.zero_grad()

                tqdmDataLoader.set_postfix(ordered_dict={
                                    "Epoch":epoch+1,
                                    "L_l1": loss_l1.item(),
                                    # "L_sm": loss_sm.item(),
                                    "LR":optim.param_groups[0]['lr'],
                                })
                list_gts += lbl.cpu().view(-1).numpy().tolist()
                list_preds += out.detach().cpu().view(-1).numpy().tolist()

        lr_sh.step()
        train_score = Score_s3(list_preds, list_gts)

        plt.figure()
        seaborn.scatterplot(x=list_preds, y=list_gts)
        t_min, t_max = np.min(list_gts), np.max(list_gts)
        # plt.xlim([t_min-5, t_max+5])
        plt.savefig(log_dir+'./PD_train.png')
        

        model.eval()
        list_gts = []
        list_preds = []
        for data in tqdm(loader_valid, ncols=150):
            img = data['img'].to(device)
            lbl = data['PD']
            st = data['stage'].to(device)
            eye = data['eye'].to(device)
            age = data['age'].to(device)

            with torch.no_grad():
                out = model.forward_pd(img, st, eye, age)
            
            
            list_gts += lbl.view(-1).numpy().tolist()
            list_preds += out.detach().cpu().view(-1).numpy().tolist()
        temp_score = Score_s3(list_preds, list_gts)

        plt.figure()
        seaborn.scatterplot(x=list_preds, y=list_gts)
        t_min, t_max = np.min(list_gts), np.max(list_gts)
        # plt.xlim([t_min-5, t_max+5])
        plt.savefig(log_dir+'./PD_valid.png')
        plt.cla()
        plt.close('all')

        print('train Score:{:.4f}'.format(train_score))
        print('valid Score:{:.4f}'.format(temp_score))

        if temp_score >= best_score:
            best_score = temp_score
            torch.save(model.state_dict() ,'./model_t3_best.pb')
            print('Best Score:{:.4f},---------------------------------------------------------good job--------------------------------------------------------'.format(temp_score))

            plt.figure()
            seaborn.scatterplot(x=list_preds, y=list_gts)
            t_min, t_max = np.min(list_gts), np.max(list_gts)
            # plt.xlim([t_min-5, t_max+5])
            plt.savefig(log_dir+'./PD_valid_best.png')
        torch.save(model.state_dict(), './model_t3.pb')




    filelists = os.listdir(test_root)
    test_filelists = []
    for f in filelists:
        if 'DS' in f:
            continue
        else:
            test_filelists.append(f)
    
    list_data = np.load('./dataset_'+str(re_size)+'_test.npy')
    print('load dataset finished~')
    dict_imgs = {}

    for f_id, oct in zip(test_filelists, list_data):
        dict_imgs[f_id] = oct



    filelists = os.listdir(test_root)
    test_dataset = dataset_sub3(dict_imgs,
                                data_info='./data/STAGE_validation/STAGE_validation/data_info_validation.xlsx',
                                mode='test')
    loader_test = DataLoader(test_dataset, 1, False, num_workers=0, drop_last=False)

    model_eval.load_state_dict(torch.load('./model_t3_best.pb', map_location='cpu'), strict=True)
    model_eval = model_eval.to(device)
    model_eval.eval()

    list_results = []
    for data in tqdm(loader_test, ncols=150):
        img = data['img'].to(device)
        idx = data['img_id']
        st = data['stage'].to(device)
        eye = data['eye'].to(device)
        age = data['age'].to(device)

 

        with torch.no_grad():
            out = model_eval.forward_pd(img, st, eye, age)

        out = out.detach().cpu().view(-1).numpy()+0.5
        out = out.astype(np.uint8)
        
        temp_results = [int(idx[0])] + out.tolist()
        list_results.append(temp_results)

    list_results.sort(key=lambda x:x[0])
    temp_cols = ['ID']
    for i in range(52):
        temp_cols.append('point'+str(i+1))
    submission_result = pd.DataFrame(list_results, columns=temp_cols)
    submission_result.to_csv("./PD_Results.csv", index=False)

import torch.autograd as autograd


def task_mt(only_eval=False):
    log_dir = './logs/'
    os.makedirs(log_dir, exist_ok=True)
    model = resnet50().to(device)
    model_eval = resnet50()
    try:
        model.load_state_dict(torch.load('./model_mt.pb', map_location='cpu'), strict=True)
        pass
    except Exception as e:
        print(e)

    if only_eval==False:
        filelists = os.listdir(trainset_root)
        train_filelists = []
        for f in filelists:
            if 'DS' in f:
                continue
            else:
                train_filelists.append(f)
        
        list_data = np.load('./dataset_'+str(re_size)+'.npy')
        print('load dataset finished~')
        dict_imgs_train = {}
        dict_imgs_valid = {}

        for i in range(len(train_filelists)):
            f_id = train_filelists[i]
            oct = list_data[i]
            if (i+1)%(1/val_ratio)==0:
                dict_imgs_valid[f_id] = oct
            else:
                dict_imgs_train[f_id] = oct


        dataset_train = dataset_mt(
                            dict_imgs=dict_imgs_train,
                            label_files=[
                                './data/STAGE_training/STAGE_training/training_GT/task1_GT_training.xlsx',
                                './data/STAGE_training/STAGE_training/training_GT/task2_GT_training.xlsx',
                                './data/STAGE_training/STAGE_training/training_GT/task3_GT_training.xlsx',
                                ],
                            data_info='./data/STAGE_training/STAGE_training/data_info_training.xlsx',
                            mode='train',
                            data_scale=10,
                            )
        loader_train = DataLoader(dataset_train, batch_size, True, num_workers=num_workers, drop_last=True, pin_memory=False)

        dataset_valid = dataset_mt(
                            dict_imgs=dict_imgs_valid,
                            label_files=[
                                './data/STAGE_training/STAGE_training/training_GT/task1_GT_training.xlsx',
                                './data/STAGE_training/STAGE_training/training_GT/task2_GT_training.xlsx',
                                './data/STAGE_training/STAGE_training/training_GT/task3_GT_training.xlsx',
                                ],
                            data_info='./data/STAGE_training/STAGE_training/data_info_training.xlsx',
                            mode='train',
                            )
        loader_valid = DataLoader(dataset_valid, batch_size, False, num_workers=num_workers, drop_last=False)

        
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
        lr_sh = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[epoch_max*2//3], gamma=0.1)
        loss_fun = nn.SmoothL1Loss()
        loss_fun_sm = SMAPELoss()
        loss_fun_r2 = R2Loss()
        loss_fun_pd = PDl1Loss()

        best_score = 0

        for epoch in range(epoch_max):
            model.train()
            list_gts_md = []
            list_preds_md= []
            list_gts_se = []
            list_preds_se= []
            list_gts_pd = []
            list_preds_pd= []

            with tqdm(loader_train, ncols=150) as tqdmDataLoader:
                for i, data in enumerate(tqdmDataLoader):

                    img = data['img'].to(device)
                    st = data['stage'].to(device)
                    eye = data['eye'].to(device)
                    age = data['age'].to(device)
                    l_md = data['MD'].to(device)
                    l_se = data['SE'].to(device)
                    l_pd = data['PD'].to(device)


                    out_md, out_se, out_pd = model.forward_mt(img, st, age)
                    b = out_md.shape[0]

                    ##MD
                    loss_l1_md = loss_fun(out_md.view(b,-1), l_md.view(b,-1))
                    # loss_sm_md = loss_fun_sm(out_md.view(b,-1), l_md.view(b,-1))
                    # loss_r2_md = loss_fun_r2(out_md.view(b,-1), l_md.view(b,-1))

                    loss_md = loss_l1_md
                    # loss_md.backward(retain_graph=True)
                    # optim.step()

                    ##SE
                    loss_l1_se = loss_fun(out_se.view(b,-1), l_se.view(b,-1))
                    # loss_sm_se = loss_fun_sm(out_se.view(b,-1), l_se.view(b,-1))
                    # loss_r2_se = loss_fun_r2(out_se.view(b,-1), l_se.view(b,-1))

                    loss_se = loss_l1_se
                    # loss_se.backward(retain_graph=True)
                    # optim.step()

                    ##PD
                    loss_l1_pd = loss_fun_pd(out_pd.view(b, -1), l_pd.view(b, -1))

                    loss_pd = loss_l1_pd
                    # loss_pd.backward(retain_graph=True)
                    # optim.step()
                    
                    # with autograd.detect_anomaly():
                    loss = loss_md+loss_se+loss_pd
                    loss.backward()
                    optim.step()
                    optim.zero_grad()

                    tqdmDataLoader.set_postfix(ordered_dict={
                                        "Epoch":epoch+1,
                                        "L_md": loss_md.item(),
                                        "L_se": loss_se.item(),
                                        "L_pd": loss_pd.item(),
                                        "LR":optim.param_groups[0]['lr'],
                                    })
                    

                    list_gts_md += l_md.cpu().view(-1).numpy().tolist()
                    list_preds_md += out_md.detach().cpu().view(-1).numpy().tolist()

                    list_gts_se += l_se.cpu().view(-1).numpy().tolist()
                    list_preds_se += out_se.detach().cpu().view(-1).numpy().tolist()

                    list_gts_pd += l_pd.cpu().view(-1).numpy().tolist()
                    list_preds_pd += out_pd.detach().cpu().view(-1).numpy().tolist()

            
            
            
            lr_sh.step()
            train_score_md, _, _ = Score(list_preds_md, list_gts_md)
            train_score_se, _, _ = Score(list_preds_se, list_gts_se)
            train_score_pd = Score_s3(list_preds_pd, list_gts_pd)


            model.eval()
            list_gts_md = []
            list_preds_md= []
            list_gts_se = []
            list_preds_se= []
            list_gts_pd = []
            list_preds_pd= []
            for data in tqdm(loader_valid, ncols=150):
                img = data['img'].to(device)
                st = data['stage'].to(device)
                eye = data['eye'].to(device)
                age = data['age'].to(device)
                l_md = data['MD'].to(device)
                l_se = data['SE'].to(device)
                l_pd = data['PD'].to(device)

                    

                with torch.no_grad():
                    out_md, out_se, out_pd = model.forward_mt(img, st, age)

                list_gts_md += l_md.cpu().view(-1).numpy().tolist()
                list_preds_md += out_md.detach().cpu().view(-1).numpy().tolist()

                list_gts_se += l_se.cpu().view(-1).numpy().tolist()
                list_preds_se += out_se.detach().cpu().view(-1).numpy().tolist()

                list_gts_pd += l_pd.cpu().view(-1).numpy().tolist()
                list_preds_pd += out_pd.detach().cpu().view(-1).numpy().tolist()


            valid_score_md, _, _ = Score(list_preds_md, list_gts_md)
            valid_score_se, _, _ = Score(list_preds_se, list_gts_se)
            valid_score_pd = Score_s3(list_preds_pd, list_gts_pd)*10.0
            
            # plt.figure()
            # seaborn.scatterplot(x=list_preds, y=list_gts)
            # t_min, t_max = np.min(list_gts), np.max(list_gts)
            # plt.xlim([t_min-5, t_max+5])
            # plt.savefig(log_dir+'./SE_valid.png')

            print('train Score MD:{:.4f}, SE:{:.4f}, PD:{:.4f}'.format(train_score_md, train_score_se, train_score_pd))
            print('valid Score MD:{:.4f}, SE:{:.4f}, PD:{:.4f}'.format(valid_score_md, valid_score_se, valid_score_pd))


            temp_score_avg = (valid_score_md+valid_score_se+valid_score_pd)/3.0
            if temp_score_avg >= best_score:
                best_score = temp_score_avg
                torch.save(model.state_dict() ,'./model_best_mt.pb')
                print('Best Score MD:{:.4f}, SE:{:.4f}, PD:{:.4f}---------------------------------------------------------good job--------------------------------------------------------'.format(valid_score_md, valid_score_se, valid_score_pd))
                # plt.figure()
                # seaborn.scatterplot(x=list_preds, y=list_gts)
                # t_min, t_max = np.min(list_gts), np.max(list_gts)
                # plt.xlim([t_min-5, t_max+5])
                # plt.savefig(log_dir+'./SE_valid_best.png')
            torch.save(model.state_dict(), './model_mt.pb')
            plt.cla()
            plt.close('all')

    
    
    filelists = os.listdir(test_root)
    test_filelists = []
    for f in filelists:
        if 'DS' in f:
            continue
        else:
            test_filelists.append(f)
    
    list_data = np.load('./dataset_'+str(re_size)+'_test.npy')
    print('load dataset finished~')
    dict_imgs = {}

    for f_id, oct in zip(test_filelists, list_data):
        dict_imgs[f_id] = oct



    filelists = os.listdir(test_root)
    test_dataset = dataset_mt(dict_imgs,
                                data_info='./data/STAGE_validation/STAGE_validation/data_info_validation.xlsx',
                                mode='test')
    loader_test = DataLoader(test_dataset, 1, False, num_workers=0, drop_last=False)

    
    model_eval.load_state_dict(torch.load('./model_best_mt.pb', map_location='cpu'), strict=True)
    model_eval = model_eval.to(device)
    model_eval.eval()

    list_results_MD = []
    list_results_SE = []
    list_results_PD = []
    for data in tqdm(loader_test, ncols=150):
        img = data['img'].to(device)
        idx = data['img_id']
        st = data['stage'].to(device)
        eye = data['eye'].to(device)
        age = data['age'].to(device)

        with torch.no_grad():
            out_md, out_se, out_pd = model_eval.forward_mt(img, st, age)



        ##MD
        pred = out_md.detach().cpu().view(-1).numpy()
        temp_results = [int(idx[0])] + pred.tolist()
        list_results_MD.append(temp_results)


        ##SE
        out = out_se[0][0].detach().cpu().numpy()
        out[out<0]=0
        out = SEmap2pts(out, eye)
        temp_results = [int(idx[0])] + out
        list_results_SE.append(temp_results)

        ##PD
        out = out_pd[0][0].detach().cpu().numpy()
        out[out<0]=0
        out = out.astype(np.uint8)
        out[out>4]=4
        out = SEmap2pts(out, eye)
        temp_results = [int(idx[0])] + out
        list_results_PD.append(temp_results)

        
    
    list_results_MD.sort(key=lambda x:x[0])
    submission_result = pd.DataFrame(list_results_MD, columns=['ID', 'pred_MD'])
    submission_result.to_csv("./MD_Results.csv", index=False)
    
    
    
    list_results_SE.sort(key=lambda x:x[0])
    temp_cols = ['ID']
    for i in range(52):
        temp_cols.append('point'+str(i+1))
    submission_result = pd.DataFrame(list_results_SE, columns=temp_cols)
    submission_result.to_csv("./Sensitivity_map_Results.csv", index=False)

    
    list_results_PD.sort(key=lambda x:x[0])
    temp_cols = ['ID']
    for i in range(52):
        temp_cols.append('point'+str(i+1))
    submission_result = pd.DataFrame(list_results_PD, columns=temp_cols)
    submission_result.to_csv("./PD_Results.csv", index=False)


def eval_task_mt(only_eval=False):
    model_eval = resnet50()
    filelists = os.listdir(test_root)
    test_filelists = []
    for f in filelists:
        if 'DS' in f:
            continue
        else:
            test_filelists.append(f)
    
    list_data = np.load('./dataset_'+str(re_size)+'_test.npy')
    print('load dataset finished~')
    dict_imgs = {}

    for f_id, oct in zip(test_filelists, list_data):
        dict_imgs[f_id] = oct



    filelists = os.listdir(test_root)
    test_dataset = dataset_mt(dict_imgs,
                                data_info='./data/STAGE_validation/STAGE_validation/data_info_validation.xlsx',
                                mode='test')
    loader_test = DataLoader(test_dataset, 1, False, num_workers=0, drop_last=False)

    
    model_eval.load_state_dict(torch.load('./model_best_mt.pb', map_location='cpu'), strict=True)
    model_eval = model_eval.to(device)
    model_eval.eval()

    list_results_MD = []
    list_results_SE = []
    list_results_PD = []
    for data in tqdm(loader_test, ncols=150):
        img = data['img'].to(device)
        idx = data['img_id']
        st = data['stage'].to(device)
        eye = data['eye'].to(device)
        age = data['age'].to(device)

        with torch.no_grad():
            out_md, out_se, out_pd = model_eval.forward_mt(img, st, age)



        ##MD
        pred = out_md.detach().cpu().view(-1).numpy()
        temp_results = [int(idx[0])] + pred.tolist()
        list_results_MD.append(temp_results)


        ##SE
        out = out_se[0][0].detach().cpu().numpy()
        out[out<0]=0
        out = SEmap2pts(out, eye)
        temp_results = [int(idx[0])] + out
        list_results_SE.append(temp_results)

        ##PD
        out = out_pd[0][0].detach().cpu().numpy()
        out[out<0]=0
        out = out.astype(np.uint8)
        out[out>4]=4
        out = SEmap2pts(out, eye)
        temp_results = [int(idx[0])] + out
        list_results_PD.append(temp_results)

        
    
    list_results_MD.sort(key=lambda x:x[0])
    submission_result = pd.DataFrame(list_results_MD, columns=['ID', 'pred_MD'])
    submission_result.to_csv("./MD_Results.csv", index=False)
    
    
    
    list_results_SE.sort(key=lambda x:x[0])
    temp_cols = ['ID']
    for i in range(52):
        temp_cols.append('point'+str(i+1))
    submission_result = pd.DataFrame(list_results_SE, columns=temp_cols)
    submission_result.to_csv("./Sensitivity_map_Results.csv", index=False)

    
    list_results_PD.sort(key=lambda x:x[0])
    temp_cols = ['ID']
    for i in range(52):
        temp_cols.append('point'+str(i+1))
    submission_result = pd.DataFrame(list_results_PD, columns=temp_cols)
    submission_result.to_csv("./PD_Results.csv", index=False)

'./data/STAGE_FINAL_IMGS/STAGE_FINAL_IMGS/data_info_testing.xlsx'
def test_task_mt(only_eval=False):
    model_eval = resnet50()

    test_dataset = dataset_mt_scratch(
                                test_root=test_root,
                                data_info='./data/STAGE_FINAL_IMGS/STAGE_FINAL_IMGS/data_info_testing.xlsx',
                                mode='test')
    loader_test = DataLoader(test_dataset, 1, False, num_workers=0, drop_last=False)

    
    model_eval.load_state_dict(torch.load('./model_best_mt.pb', map_location='cpu'), strict=True)
    model_eval = model_eval.to(device)
    model_eval.eval()

    list_results_MD = []
    list_results_SE = []
    list_results_PD = []
    for data in tqdm(loader_test, ncols=150):
        img = data['img'].to(device)
        idx = data['img_id']
        st = data['stage'].to(device)
        eye = data['eye'].to(device)
        age = data['age'].to(device)

        with torch.no_grad():
            out_md, out_se, out_pd = model_eval.forward_mt(img, st, age)



        ##MD
        pred = out_md.detach().cpu().view(-1).numpy()
        temp_results = [int(idx[0])] + pred.tolist()
        list_results_MD.append(temp_results)


        ##SE
        out = out_se[0][0].detach().cpu().numpy()
        out[out<0]=0
        out = SEmap2pts(out, eye)
        temp_results = [int(idx[0])] + out
        list_results_SE.append(temp_results)

        ##PD
        out = out_pd[0][0].detach().cpu().numpy()
        out[out<0]=0
        out = out.astype(np.uint8)
        out[out>4]=4
        out = SEmap2pts(out, eye)
        temp_results = [int(idx[0])] + out
        list_results_PD.append(temp_results)

        
    
    list_results_MD.sort(key=lambda x:x[0])
    submission_result = pd.DataFrame(list_results_MD, columns=['ID', 'pred_MD'])
    submission_result.to_csv("./MD_Results.csv", index=False)
    
    
    
    list_results_SE.sort(key=lambda x:x[0])
    temp_cols = ['ID']
    for i in range(52):
        temp_cols.append('point'+str(i+1))
    submission_result = pd.DataFrame(list_results_SE, columns=temp_cols)
    submission_result.to_csv("./Sensitivity_map_Results.csv", index=False)

    
    list_results_PD.sort(key=lambda x:x[0])
    temp_cols = ['ID']
    for i in range(52):
        temp_cols.append('point'+str(i+1))
    submission_result = pd.DataFrame(list_results_PD, columns=temp_cols)
    submission_result.to_csv("./PD_Results.csv", index=False)



from datasets import dict_stage
import matplotlib.pyplot as plt
import seaborn

def analysis_data_MD():
    md_dir = './data/STAGE_training/STAGE_training/training_GT/task1_GT_training.xlsx'
    info_dir = './data/STAGE_training/STAGE_training/data_info_training.xlsx'
    label = {}
    for _, row in pd.read_excel(md_dir).iterrows():
        label[int((row['ID']))]=row[1]
    info = {}
    for _, row in pd.read_excel(info_dir).iterrows():
        temp_info_gender = row[1]
        temp_info_age = row[2]
        temp_info_eye = row[3]
        temp_info_st =  dict_stage[row[4]]
        info[int((row['ID']))]=[temp_info_gender, temp_info_age, temp_info_eye, temp_info_st]
    
    list_md = []
    list_st = []
    list_age = []

    for k,v in label.items():
        list_md.append(v)
        list_st.append(info[k][3])
        list_age.append(info[k][1])
    
    plt.figure()
    seaborn.scatterplot(list_md, list_st)
    plt.savefig('./MD_stage.png')
    plt.figure()
    seaborn.scatterplot(list_md, list_age)
    plt.savefig('./MD_age.png')
    # plt.show()


def analysis_data_SE():
    md_dir = './data/STAGE_training/STAGE_training/training_GT/task2_GT_training.xlsx'
    info_dir = './data/STAGE_training/STAGE_training/data_info_training.xlsx'
    label = {}
    for _, row in pd.read_excel(md_dir).iterrows():
        label[int((row['ID']))]=row[1:].values.tolist()
    info = {}
    for _, row in pd.read_excel(info_dir).iterrows():
        temp_info_gender = row[1]
        temp_info_age = row[2]
        temp_info_eye = row[3]
        temp_info_st =  dict_stage[row[4]]
        info[int((row['ID']))]=[temp_info_gender, temp_info_age, temp_info_eye, temp_info_st]
    
    list_se = []


    for k,v in label.items():
        list_se += v

    
    plt.figure()
    seaborn.histplot(list_se)
    plt.savefig('./SE_hist.png')

    # plt.show()


def analysis_data_PD():
    md_dir = './data/STAGE_training/STAGE_training/training_GT/task3_GT_training.xlsx'
    info_dir = './data/STAGE_training/STAGE_training/data_info_training.xlsx'
    label = {}
    for _, row in pd.read_excel(md_dir).iterrows():
        label[int((row['ID']))]=row[1:].values.tolist()
    info = {}
    for _, row in pd.read_excel(info_dir).iterrows():
        temp_info_gender = row[1]
        temp_info_age = row[2]
        temp_info_eye = row[3]
        temp_info_st =  dict_stage[row[4]]
        info[int((row['ID']))]=[temp_info_gender, temp_info_age, temp_info_eye, temp_info_st]
    
    list_pd = []


    for k,v in label.items():
        list_pd += v

    
    plt.figure()
    seaborn.histplot(list_pd)
    plt.savefig('./PD_hist.png')

    # plt.show()



if __name__ == '__main__':

    ## convert data from image slices to 3D-numpy file
    # generate_dataset_train(512)
    # generate_dataset_test(512)

    
    ## analysis data values
    # analysis_data_MD()
    # analysis_data_SE()
    # analysis_data_PD()

    
    ## base config
    batch_size = 8
    lr = 1e-4
    num_workers = 2
    val_ratio = 0.2
    epoch_max = 100
    re_size = 512


    ## train and eval mt framework of task1,2,3
    # task_mt()

    ## eval mt framework of task 1, 2, 3 (need data preprocess from imageFile to numpy)
    # eval_task_mt()

    ## test mt framework of task 1, 2, 3(data from scratch)
    test_task_mt()