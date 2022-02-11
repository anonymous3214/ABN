from model import *
from utils import *


def train_2d(img_size,
          num_stage,
          train_set_name,
          test_set_name,
          batch_size,
          num_epochs,
          model_name,
          loss_name,
          smooth_name,
          learning_rate,
          lamda,
          penalty,
          save_every_epoch,
          num_sample,
          num_lines,
          grid_sample_orig):
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = model_name.to(device)
    smooth = smooth_name.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ssim=SSIM().to(device)
    CC=GCC().to(device)
    
    if loss_name=="GCC":
        loss_func=GCC().to(device)
    elif loss_name=="LCC":
        loss_func=LCC().to(device)
    elif loss_name=="MSE":
        loss_func=nn.MSELoss().to(device)
    elif loss_name=="NCC":
        loss_func=NCC().loss
    elif loss_name=="SSIM":
        loss_func=SSIM().to(device)
    
    cur_path = os.getcwd()
    result_path=cur_path+'/result'
    
    loss_log_path=result_path+'/loss_log'
    create_folder(result_path,'loss_log')
    
    sample_img_path=result_path+'/sample_img'
    create_folder(result_path,'sample_img')
    
    model_save_path=result_path+'/model'
    create_folder(result_path,'model')
    
    model_str=str(model)[0:str(model).find("(")]
    smooth_str=str(smooth)[0:str(smooth).find("(")]
    lamda_str=str(lamda)
    dataset_str=train_set_name[0:str(train_set_name).find("_")]
    num_stage_str=str(num_stage)
    
    modal_name=model_str+"_"+num_stage_str+"_"+loss_name+"_"+smooth_str+"_"+penalty+"_λ_"+lamda_str+"_"+dataset_str
    modal_path=sample_img_path+"/"+modal_name
    create_folder(sample_img_path,modal_name)
    
    sample_o_path=modal_path+"/"+"o"
    sample_t_path=modal_path+"/"+"t"
    create_folder(modal_path,"o")
    create_folder(modal_path,"t")
    
    for i in range(int(num_stage)):
        idx=i+1
        p_name="p_"+str(idx)
        p_grid_name="p_"+str(idx)+"_grid"
        sample_p_path=modal_path+"/"+p_name
        sample_p_grid_path=modal_path+"/"+p_grid_name
        
        create_folder(modal_path,p_name)
        create_folder(modal_path,p_grid_name)
    

    modal_info="Model: {}    num_stage: {}    Loss: {}    Smooth: {}    Penalty: {}    λ: {}    dataset: {}".format(model_str,
                                                                                       num_stage_str,
                                                                                       loss_name,
                                                                                       smooth_str,
                                                                                       penalty,
                                                                                       lamda_str,
                                                                                       dataset_str)
    create_log(modal_info,loss_log_path,modal_name)
    
    print (modal_info)
    train_loader=load_data(train_set_name,batch_size)
    test_loader=load_data(test_set_name,batch_size)
    
    for epoch in range(num_epochs):
        
        total_loss_train=[]
        total_mse_loss_train=[]
        total_ssim_loss_train=[]
        
        start=time.time()
        for i, x in enumerate(train_loader):
            t_data,o_data,t_data_gray,o_data_gray,grid=x
            
            t_data=t_data.to(device).view(-1,3,img_size,img_size)
            o_data=o_data.to(device).view(-1,3,img_size,img_size)
            
            t_data_gray=t_data_gray.to(device).view(-1,1,img_size,img_size)
            o_data_gray=o_data_gray.to(device).view(-1,1,img_size,img_size)
            
            y_grid=grid.to(device).view(-1,img_size,img_size,2)
            
            
            optimizer.zero_grad()
            outputs_L,flow_L,p_grid_L=model(o_data,t_data)
            
            mse_loss=loss_func(outputs_L[-1],o_data)
            ssim_loss=ssim(outputs_L[-1],o_data)
            loss=mse_loss+lamda*sum([smooth(i) for i in flow_L])
            
            
            loss.backward()
            optimizer.step()

            ground_truth_loss=F.mse_loss(p_grid_L[-1],y_grid)
            
            
            total_loss_train.append(loss.item())
            total_mse_loss_train.append(mse_loss.item())
            total_ssim_loss_train.append(ssim_loss.item())
            
        ave_loss_train=torch.mean(torch.FloatTensor(total_loss_train))
        std_loss_train=torch.std(torch.FloatTensor(total_loss_train))
        ave_mse_loss_train=torch.mean(torch.FloatTensor(total_mse_loss_train))
        std_mse_loss_train=torch.std(torch.FloatTensor(total_mse_loss_train))
        ave_ssim_loss_train=torch.mean(torch.FloatTensor(total_ssim_loss_train))
        std_ssim_loss_train=torch.std(torch.FloatTensor(total_ssim_loss_train))
            

        if epoch % save_every_epoch ==0:
            model.eval()
            with torch.no_grad():
                total_mse_loss_test=[]
                total_ssim_loss_test=[]
                total_cc_loss_test=[]

                for i, x in enumerate(test_loader):
                    t_data,o_data,t_data_gray,o_data_gray,grid=x

                    t_data=t_data.to(device).view(-1,3,img_size,img_size)
                    o_data=o_data.to(device).view(-1,3,img_size,img_size)

                    t_data_gray=t_data_gray.to(device).view(-1,1,img_size,img_size)
                    o_data_gray=o_data_gray.to(device).view(-1,1,img_size,img_size)

                    y_grid=grid.to(device).view(-1,img_size,img_size,2)

                    outputs_L,flow_L,p_grid_L=model(o_data,t_data)

                    #loss=F.mse_loss(outputs,o_data)
                    loss=loss_func(outputs_L[-1],o_data)
                    ssim_loss=ssim(outputs_L[-1],o_data)

                    
                    cc_loss=CC(outputs_L[-1],o_data)
                    
                    total_mse_loss_test.append(loss.item())
                    total_ssim_loss_test.append(ssim_loss.item())
                    total_cc_loss_test.append(cc_loss)

                ave_mse_loss_test=torch.mean(torch.FloatTensor(total_mse_loss_test))
                std_mse_loss_test=torch.std(torch.FloatTensor(total_mse_loss_test))
                ave_ssim_loss_test=torch.mean(torch.FloatTensor(total_ssim_loss_test))
                std_ssim_loss_test=torch.std(torch.FloatTensor(total_ssim_loss_test))
                ave_cc_loss_test=torch.mean(torch.FloatTensor(total_cc_loss_test))
                std_cc_loss_test=torch.std(torch.FloatTensor(total_cc_loss_test))
        

                loss_info="Epoch[{}/{}], All Training loss: {:.4f}/{:.4f} , MSE Training loss: {:.4f}/{:.4f} , SSIM Training loss: {:.4f}/{:.4f}  ,  MSE test loss: {:.4f}/{:.4f}  ,  SSIM test loss: {:.4f}/{:.4f}  ,  CC test loss: {:.4f}/{:.4f}".format(epoch+1,num_epochs,
                ave_loss_train,std_loss_train,
                ave_mse_loss_train,std_mse_loss_train,
                ave_ssim_loss_train,std_ssim_loss_train,
                ave_mse_loss_test,std_mse_loss_test,
                ave_ssim_loss_test,std_ssim_loss_test,
                ave_cc_loss_test,std_cc_loss_test)
            
            
                print (loss_info)
                append_log(loss_info,loss_log_path,modal_name)
                
                
                
                #####save image##
                
                save_sample_any(epoch,"o",o_data,sample_o_path,num_sample)
                save_sample_any(epoch,"t",t_data,sample_t_path,num_sample)


                batch_size=outputs_L[0].shape[0]
                image_grid_prev=grid_i_plot(img_size,batch_size,num_lines=num_lines)

                for i in range(int(num_stage)):
                    idx=i+1
                    p_name="p_"+str(idx)
                    p_grid_name="p_"+str(idx)+"_grid"
                    sample_p_path=modal_path+"/"+p_name
                    sample_p_grid_path=modal_path+"/"+p_grid_name

                    image_grid_cur,final_img=grid_plot(image_grid_prev,p_grid_L[i],outputs_L[i])
                    
                    if grid_sample_orig==False:
                        image_grid_prev=image_grid_cur
                    else:
                        image_grid_prev=image_grid_prev


                    save_sample_any(epoch,p_name,outputs_L[i],sample_p_path,num_sample)
                    save_sample_any(epoch,p_grid_name,final_img,sample_p_grid_path,num_sample)
                                
            torch.save(model.state_dict(), os.path.join(model_save_path,modal_name+"_"+str(epoch)+".pth"))
    return