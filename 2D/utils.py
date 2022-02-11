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
from math import exp


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx][0],self.data[idx][1],self.data[idx][2],self.data[idx][3],self.data[idx][4]

def create_folder(path,folder_name):
    folder_path=path+'/'+folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
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

def create_log(log,path,file_name):
    with open(path+"/"+file_name+".txt", 'w') as output:
        output.write(str(log) + '\n')
    return

def append_log(log,path,file_name):
    with open(path+"/"+file_name+".txt", 'a+') as output:
        output.write(str(log) + '\n')
    return

class GCC(nn.Module):
    def __init__(self):
        super(GCC, self).__init__()
 
    def forward(self, I, J):
        I2 = I.pow(2)
        J2 = J.pow(2)
        IJ = I * J

        I_ave, J_ave= I.mean(), J.mean()
        I2_ave, J2_ave = I2.mean(), J2.mean()
        IJ_ave = IJ.mean()
        
        cross = IJ_ave - I_ave * J_ave
        I_var = I2_ave - I_ave.pow(2)
        J_var = J2_ave - J_ave.pow(2)
        cc = cross / (I_var.sqrt() * J_var.sqrt() + np.finfo(float).eps)
        return 1.0 * cc 

class LCC(nn.Module):
    def __init__(self, win=[9, 9], eps=1e-5):
        
        super(LCC, self).__init__()
        self.win = win
        self.eps = eps
        
    def forward(self, I, J):
        I2 = I.pow(2)
        J2 = J.pow(2)
        IJ = I * J
        
        filters = Variable(torch.ones(1, 1, self.win[0], self.win[1]))
        if I.is_cuda:
            filters = filters.cuda()
        padding = (self.win[0]//2, self.win[1]//2)
        
        I_sum = F.conv2d(I, filters, stride=1, padding=padding)
        J_sum = F.conv2d(J, filters, stride=1, padding=padding)
        I2_sum = F.conv2d(I2, filters, stride=1, padding=padding)
        J2_sum = F.conv2d(J2, filters, stride=1, padding=padding)
        IJ_sum = F.conv2d(IJ, filters, stride=1, padding=padding)
        
        win_size = self.win[0]*self.win[1]
 
        u_I = I_sum / win_size
        u_J = J_sum / win_size
        
        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size
 
        cc = cross*cross / (I_var*J_var + self.eps)#np.finfo(float).eps
        lcc = -1.0 * torch.mean(cc) + 1
        return lcc


class second_Grad(nn.Module):

    def __init__(self, penalty):
        super(second_Grad, self).__init__()
        self.penalty = penalty
    
    def forward(self, pred):
        
        dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]) 
        dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])  
       
        dyy = torch.abs(dy[:, :, 1:, :] - dy[:, :, :-1, :])
        dyx = torch.abs(dy[:, :, :, 1:] - dy[:, :, :, :-1])
        
        dxx = torch.abs(dx[:, :, :, 1:] - dx[:, :, :, :-1])
        dxy = torch.abs(dx[:, :, 1:, :] - dx[:, :, :-1, :])
        
        if self.penalty == 'l2':
            dyy = dyy * dyy
            dyx = dyx * dyx

            dxx = dxx * dxx
            dxy = dxy * dxy
        elif self.penalty == 'l1':
            dyy = dyy
            dyx = dyx 

            dxx = dxx
            dxy = dxy
        
        d = torch.mean(dyy) + torch.mean(dyx) + torch.mean(dxx) + torch.mean(dxy)
        grad = d / 4.0
        
        return grad  

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()
    
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window  

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)
    

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)


def save_sample_any(epoch,o_name,o_data,o_path,num_sample):
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    
    sample_o=o_data[0:num_sample]
    
    o_name_img=o_name+"_"+str(epoch)+".png"

    image_o=np.transpose(vutils.make_grid(sample_o.to(device), nrow=int(np.sqrt(num_sample)), normalize=True).cpu(),(1,2,0)).numpy()
    matplotlib.image.imsave(o_path+"/"+o_name_img, image_o,dpi=600)
    
    return

def grid_plot(image_grid_prev,df_grid,image_obj_cur):
    image_grid_cur=F.grid_sample(image_grid_prev,df_grid,padding_mode='zeros',align_corners=True)
    image_obj_cur_3ch=image_obj_cur
    final_img=image_grid_cur+image_obj_cur_3ch
    return image_grid_cur,final_img

def grid_i_plot(img_size,batch_size,num_lines):
    image=np.zeros((img_size,img_size,3))
    skip=img_size//num_lines
    idx_L=list(range(0, image.shape[1],skip))
    image=image.transpose((2,0,1))
    for i in idx_L[1:]:
        image[0][i]=1
        image[0:1,:,i]=1
    image=image.transpose((1,2,0)).astype(np.float)
    image_th=torch.from_numpy(image).float().cuda()
    image_th=image_th.unsqueeze(0).permute(0,3,1,2)
    image_th_batch = image_th.repeat(batch_size, 1, 1, 1)
    return image_th_batch