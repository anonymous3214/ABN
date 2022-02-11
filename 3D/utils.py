import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.autograd import Variable
import matplotlib
import SimpleITK as sitk
import math


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx][0],self.data[idx][1],self.data[idx][2],self.data[idx][3]


def create_folder(path,folder_name):
    folder_path=path+'/'+folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return

def create_log(log,path,file_name):
    with open(path+"/"+file_name+".txt", 'w') as output:
        output.write(str(log) + '\n')
    return

def append_log(log,path,file_name):
    with open(path+"/"+file_name+".txt", 'a+') as output:
        output.write(str(log) + '\n')
    return

def load_data(set_name,batch_size,ifshuffle=True):
    
    cur_path = os.getcwd()
    set_path=cur_path+'/dataset/'+set_name
  
    data=np.load(set_path,allow_pickle=True)
    data_set = MyDataset(data)
    loader = torch.utils.data.DataLoader(dataset=data_set,
                                          batch_size=batch_size, 
                                          shuffle=ifshuffle)
    return loader



def save_sample_any(epoch,img_name,img,img_path):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    torch_img=img.squeeze()
    x,y,z=torch_img.shape

    torch_lr=torch_img.permute(0,1,2)
    torch_lr=torch_lr.view(x,1,y,z)

    torch_fb=torch_img.permute(2,0,1)
    torch_fb=torch_fb.view(y,1,x,z)
    torch_fb=torch_fb.permute(0, 1, 2,3).flip(2)

    torch_td=torch_img.permute(1,0,2)
    torch_td=torch_td.view(z,1,x,y)
    torch_td=torch_td.permute(0, 1, 2,3).flip(2)
    
    cat_image=torch.cat((torch_lr[x//2], torch_fb[y//2], torch_td[z//2]))
    cat_image=cat_image.view(3,1,x,y)
    
    name_img=img_name+"_"+str(epoch)+".png"
    image_o=np.transpose(vutils.make_grid(cat_image.to(device), nrow=3, normalize=True).cpu(),(1,2,0)).numpy()
    
    matplotlib.image.imsave(img_path+"/"+name_img,image_o)
    
    return 

def save_nii_any(epoch,img_name,img,img_path):
    ref_img_GetOrigin=(0.0, 0.0, 0.0)
    ref_img_GetDirection=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    ref_img_GetSpacing=(1.0, 1.0, 1.0)

    img = sitk.GetImageFromArray(img.squeeze().cpu().detach().numpy())
    
    img.SetOrigin(ref_img_GetOrigin)
    img.SetDirection(ref_img_GetDirection)
    img.SetSpacing(ref_img_GetSpacing)
    
    name_img=img_name+"_"+str(epoch)+".nii.gz"
    
    sitk.WriteImage(img, img_path+"/"+name_img)
    
    return

class second_Grad(nn.Module):

    def __init__(self, penalty):
        super(second_Grad, self).__init__()
        self.penalty = penalty
    
    def forward(self, pred):
        
        dy = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]) 
        dx = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]) 
        dz = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]) 
        

        dyy = torch.abs(dy[:, :, 1:, :, :] - dy[:, :, :-1, :, :]) 
        dyx = torch.abs(dy[:, :, :, 1:, :] - dy[:, :, :, :-1, :]) 
        dyz = torch.abs(dy[:, :, :, :, 1:] - dy[:, :, :, :, :-1])
        
        dxy = torch.abs(dx[:, :, 1:, :, :] - dx[:, :, :-1, :, :]) 
        dxx = torch.abs(dx[:, :, :, 1:, :] - dx[:, :, :, :-1, :]) 
        dxz = torch.abs(dx[:, :, :, :, 1:] - dx[:, :, :, :, :-1])
        
        dzy = torch.abs(dz[:, :, 1:, :, :] - dz[:, :, :-1, :, :]) 
        dzx = torch.abs(dz[:, :, :, 1:, :] - dz[:, :, :, :-1, :]) 
        dzz = torch.abs(dz[:, :, :, :, 1:] - dz[:, :, :, :, :-1])
        
        
        if self.penalty == 'l2':
            dyy = dyy * dyy
            dyx = dyx * dyx
            dyz = dyz * dyz

            dxy = dxy * dxy
            dxx = dxx * dxx
            dxz = dxz * dxz
            
            dzy = dzy * dzy
            dzx = dzx * dzx
            dzz = dzz * dzz

        elif self.penalty == 'l1':
            dyy = dyy 
            dyx = dyx 
            dyz = dyz 

            dxy = dxy 
            dxx = dxx 
            dxz = dxz 
            
            dzy = dzy 
            dzx = dzx 
            dzz = dzz 
        
        d = torch.mean(dyy) + torch.mean(dyx) + torch.mean(dyz) + torch.mean(dxy) + torch.mean(dxx) + torch.mean(dxz)+torch.mean(dzy) + torch.mean(dzx) + torch.mean(dzz)
        grad = d / 9.0
        
        return grad

class NCC:

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        I = y_true
        J = y_pred

        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        win = [9] * ndims if self.win is None else self.win

        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0]/2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)

        conv_fn = getattr(F, 'conv%dd' % ndims)

        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)    


def DSC(pred, target):
    smooth = 1e-5
    m1 = pred.flatten()
    m2 = target.flatten()
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def compute_label_dice(gt, pred):
    gt=gt.squeeze()
    pred=pred.squeeze()
    cls_lst = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62,
               63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161, 162,
               163, 164, 165, 166]
    dice_lst = []
    for cls in cls_lst:
        dice = DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    return torch.mean(torch.FloatTensor(dice_lst))

