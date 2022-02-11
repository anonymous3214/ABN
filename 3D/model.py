import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Warper3d(nn.Module):
    def __init__(self, img_size, model):
        super(Warper3d, self).__init__()
        
        self.model = model
        self.img_size = img_size
        D, H, W = img_size, img_size, img_size

        xx = torch.arange(0, W).view(1,1,-1).repeat(D,H,1).view(1,D,H,W)
        yy = torch.arange(0, H).view(1,-1,1).repeat(D,1,W).view(1,D,H,W)
        zz = torch.arange(0, D).view(-1,1,1).repeat(1,H,W).view(1,D,H,W)
        self.grid = torch.cat((xx,yy,zz),0).float() # [3, D, H, W]
            
    def forward(self, flow, img):
        grid = self.grid.repeat(flow.shape[0],1,1,1,1)
        if img.is_cuda:
            grid = grid.cuda()

        vgrid = Variable(grid, requires_grad = False) + flow

        shape = flow.shape[2:]
        for i in range(len(shape)):
            vgrid[:, i, ...] = 2 * (vgrid[:, i, ...] / (shape[i] - 1) - 0.5)
        vgrid = vgrid.permute(0,2,3,4,1)        
        output = F.grid_sample(img, vgrid,mode=self.model,align_corners=True)
        
        return output,vgrid
    
class conv_block_3D(nn.Module):
    def __init__(self, inChan, outChan, stride=1):
        super(conv_block_3D, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv3d(inChan, outChan, kernel_size=3, stride=stride, padding=1, bias=True),
                nn.BatchNorm3d(outChan),
                nn.LeakyReLU(0.2, inplace=True)
                )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, x):
        x = self.conv(x)

        return x
    
   
class Unet_3D(nn.Module):
    def __init__(self, enc_nf=[2,16,32,32,64,64], dec_nf=[64,32,32,32,16,3]):
        super(Unet_3D, self).__init__()
        self.inconv = conv_block_3D(enc_nf[0], enc_nf[1])
        self.down1 = conv_block_3D(enc_nf[1], enc_nf[2], 2)
        self.down2 = conv_block_3D(enc_nf[2], enc_nf[3], 2)
        self.down3 = conv_block_3D(enc_nf[3], enc_nf[4], 2)
        self.down4 = conv_block_3D(enc_nf[4], enc_nf[5], 2)
        self.up1 = conv_block_3D(enc_nf[-1], dec_nf[0])
        self.up2 = conv_block_3D(dec_nf[0]+enc_nf[4], dec_nf[1])
        self.up3 = conv_block_3D(dec_nf[1]+enc_nf[3], dec_nf[2])
        self.up4 = conv_block_3D(dec_nf[2]+enc_nf[2], dec_nf[3])
        self.same_conv = conv_block_3D(dec_nf[3]+enc_nf[1], dec_nf[4])
        self.outconv = nn.Conv3d(
                dec_nf[4], dec_nf[5], kernel_size=3, stride=1, padding=1, bias=True)
        self.outconv.weight.data.normal_(mean=0, std=1e-5)
        if self.outconv.bias is not None:
            self.outconv.bias.data.zero_()

    def forward(self, x):
        skip1 = self.inconv(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        skip4 = self.down3(skip3)
        x = self.down4(skip4)
        x = self.up1(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat((x, skip4), 1)
        x = self.up2(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat((x, skip3), 1)
        x = self.up3(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat((x, skip2), 1)
        x = self.up4(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat((x, skip1), 1)
        x = self.same_conv(x)
        x = self.outconv(x)
        
        return x    
    
    
class ABN_L_3D(nn.Module):
    def __init__(self, img_size, num_stage , modified=True, 
             enc_nf=[5,32,64,64,128,128], dec_nf=[128,64,64,64,32,3],):
        super(ABN_L_3D, self).__init__()
        self.unet = Unet_3D(enc_nf, dec_nf)
        self.warper = Warper3d(img_size,"bilinear")
        
        self.num_stage=num_stage
       
    def forward(self, ref, mov):
        img_size=ref.shape[-1]
        batch_size=ref.shape[0]

        warped_list=[]
        flow_list=[]
        grid_list=[]
        
        flow_previous= Variable(torch.zeros((batch_size, 3,img_size,img_size,img_size),dtype=torch.float)).cuda()
        cur_mov=mov
        for i in range(self.num_stage):
            image = torch.cat((ref, cur_mov,flow_previous), 1)
            cur_flow = self.unet(image)
            cur_mov,cur_grid = self.warper(cur_flow, mov)

            flow_previous=cur_flow

            warped_list.append(cur_mov)
            flow_list.append(cur_flow)
            grid_list.append(cur_grid)
                   
        return warped_list,flow_list,grid_list
    
    
class ABN_3D(nn.Module):
    def __init__(self, img_size, num_stage , modified=True, 
                 enc_nf_1=[2,16,32,32,64,64], dec_nf_1=[64,32,32,32,16,3],
                 enc_nf_2=[6,16,32,32,64,64], dec_nf_2=[64,32,32,32,16,3]):
        super(ABN_3D, self).__init__()
        self.unet_1 = Unet_3D(enc_nf_1, dec_nf_1)
        self.unet_2 = Unet_3D(enc_nf_2, dec_nf_2)
        self.warper = Warper3d(img_size,"bilinear")
        self.num_stage=num_stage

    def forward(self, ref, mov):
        img_size=ref.shape[-1]
        batch_size=ref.shape[0]
        
        warped_list=[]
        flow_list=[]
        grid_list=[]
        
        flow_previous= Variable(torch.zeros((batch_size, 3,img_size,img_size,img_size),dtype=torch.float)).cuda()
        cur_mov=mov
        
        for i in range(self.num_stage):
            image = torch.cat((ref, cur_mov), 1)
            flow_img = self.unet_1(image)
            
            flow_cat=torch.cat((flow_img, flow_previous), 1)
            
            flow_out=self.unet_2(flow_cat)

            cur_mov,cur_grid = self.warper(flow_out, mov)
            
            flow_previous=flow_out

            warped_list.append(cur_mov)
            flow_list.append(flow_out)
            grid_list.append(cur_grid)
                   
        return warped_list,flow_list,grid_list
    
    
   