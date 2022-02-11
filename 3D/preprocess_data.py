import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from skimage import feature
from skimage import transform
import os
import SimpleITK as sitk

def image_to_square(img):
    side=max(img.shape)
    img_new=np.zeros((side,side,side))
    az,ax,ay=(side-img.shape[0])//2,(side-img.shape[1])//2,(side-img.shape[2])//2
    img_new[az:img.shape[0]+az,ax:img.shape[1]+ax,ay:ay+img.shape[2]] = img
    return img_new


def resize(img,inter,size):
    i=torch.tensor([[[1., 0., 0., 0.],
         [0., 1., 0., 0.],
         [0., 0., 1., 0.]]], device='cuda:0')
    
    img_ex_1=np.expand_dims(img, axis=0)
    img_ex_2=np.expand_dims(img_ex_1, axis=0)

    o_data = torch.from_numpy(img_ex_2).cuda().float()
    
    grid_identity = F.affine_grid(i, torch.Size([1 ,1, size,size,size]),align_corners=True)
    img_resized=F.grid_sample(o_data,
                         grid_identity,
                         mode=inter,
                         align_corners=True,
                         padding_mode="zeros")
    
    img_resized_np=img_resized.squeeze(0).squeeze(0).cpu().detach().numpy()
    return img_resized_np
        
def get_train_test(size):
    train_range=[11,40]
    test_range=['02','03','04','05','06','07','08','09','10']
    
    train_set=[]
    test_set=[]
    
    fixed_img=sitk.ReadImage('./dataset/LPBA40/S01.delineation.skullstripped.nii.gz')
    fixed_img_np = sitk.GetArrayFromImage(fixed_img)
    fixed_img_square = image_to_square(fixed_img_np)
    fixed_img_resize = resize(fixed_img_square,"bilinear",size)
    
    
    fixed_label=sitk.ReadImage('./dataset/LPBA40/S01.delineation.structure.label.nii.gz')
    fixed_label_np = sitk.GetArrayFromImage(fixed_label)
    fixed_label_square = image_to_square(fixed_label_np)
    fixed_label_resize = resize(fixed_label_square,"nearest",size)
    
    for i in range(train_range[0],train_range[1]+1):
        train_img_name='S'+str(i)+'.delineation.skullstripped.nii.gz'
        train_label_name='S'+str(i)+'.delineation.structure.label.nii.gz'
        
        train_img=sitk.ReadImage('./dataset/LPBA40/train/'+train_img_name)
        train_label=sitk.ReadImage('./dataset/LPBA40/train_label/'+train_label_name)
        
        train_img_np = sitk.GetArrayFromImage(train_img)
        train_label_np = sitk.GetArrayFromImage(train_label)
        
        train_img_square = image_to_square(train_img_np)
        train_label_square = image_to_square(train_label_np)
        
        train_img_resize = resize(train_img_square,"bilinear",size)
        train_label_resize = resize(train_label_square,"nearest",size)
        
        
        train_set.append((fixed_img_resize,train_img_resize,fixed_label_resize,train_label_resize))
        #train_set.append((fixed_img_square,train_img_square))
        #train_set.append(train_img_resize)
        
    for j in test_range:
        test_img_name='S'+str(j)+'.delineation.skullstripped.nii.gz'
        test_label_name='S'+str(j)+'.delineation.structure.label.nii.gz'
        
        test_img=sitk.ReadImage('./dataset/LPBA40/test/'+test_img_name)
        test_label=sitk.ReadImage('./dataset/LPBA40/test_label/'+test_label_name)
        
        test_img_np = sitk.GetArrayFromImage(test_img)
        test_label_np = sitk.GetArrayFromImage(test_label)
        
        test_img_square = image_to_square(test_img_np)
        test_label_square = image_to_square(test_label_np)
        
        test_img_resize = resize(test_img_square,"bilinear",size)
        test_label_resize = resize(test_label_square,"nearest",size)
        
        test_set.append((fixed_img_resize,test_img_resize,fixed_label_resize,test_label_resize))
        #test_set.append((fixed_img_square,test_img_square))
        #test_set.append(test_img_resize)
        
    np.save('./dataset/LPBA40_train_'+str(size)+'.npy', train_set)
    np.save('./dataset/LPBA40_test_'+str(size)+'.npy', test_set)
        
    return 