from model import *
from utils import *


def train(img_size,
          sqe_len,
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
          sample_orig):
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = model_name.to(device)
    smooth = smooth_name.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    if loss_name=="NCC":
        loss_func=NCC().loss
    elif loss_name=="MSE":
        loss_func=nn.MSELoss().to(device)
    
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
    sqe_len_str=str(sqe_len)
    
    modal_name=model_str+"_"+sqe_len_str+"_"+loss_name+"_"+smooth_str+"_"+penalty+"_λ_"+lamda_str+"_"+dataset_str
    modal_path=sample_img_path+"/"+modal_name
    create_folder(sample_img_path,modal_name)
    
    sample_o_path=modal_path+"/"+"o"
    sample_t_path=modal_path+"/"+"t"
    create_folder(modal_path,"o")
    create_folder(modal_path,"t")
    
    for i in range(int(sqe_len)):
        idx=i+1
        p_name="p_"+str(idx)
        p_grid_name="p_"+str(idx)+"_grid"
        sample_p_path=modal_path+"/"+p_name
        sample_p_grid_path=modal_path+"/"+p_grid_name
        
        create_folder(modal_path,p_name)
        create_folder(modal_path,p_grid_name)
    

    modal_info="Model: {}    Sqe_len: {}    Loss: {}    Smooth: {}    Penalty: {}    λ: {}    dataset: {}".format(model_str,
                                                                                       sqe_len_str,
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
        
        #if epoch >lr_decay_point:
            #if epoch % lr_decay_every ==0:
                #for p in optimizer.param_groups:
                    #p['lr'] *= 0.9
        

        total_loss_train=[]
        total_sim_loss_train=[]
        
        start=time.time()
        for i, x in enumerate(train_loader):
            o_data,t_data,o_label,t_label=x
            
            t_data=t_data.to(device).view(-1,1,img_size,img_size,img_size).float()
            o_data=o_data.to(device).view(-1,1,img_size,img_size,img_size).float()
                 
            optimizer.zero_grad()
            outputs_L,flow_L,grid_L=model(o_data,t_data)
            
            sim_loss=loss_func(outputs_L[-1],o_data)
            loss=sim_loss+lamda*sum([smooth(i) for i in flow_L])
            
            loss.backward()
            optimizer.step()
             
            total_loss_train.append(loss.item())
            total_sim_loss_train.append(sim_loss.item())
            
        ave_loss_train=torch.mean(torch.FloatTensor(total_loss_train))
        std_loss_train=torch.std(torch.FloatTensor(total_loss_train))
        ave_sim_loss_train=torch.mean(torch.FloatTensor(total_sim_loss_train))
        std_sim_loss_train=torch.std(torch.FloatTensor(total_sim_loss_train))
        
        if epoch % save_every_epoch ==0:
            model.eval()
            with torch.no_grad():
                total_sim_loss_test=[]
                total_dice_loss=[]
                for i, x in enumerate(test_loader):
                    
                    o_data,t_data,o_label,t_label=x
        
                    t_data=t_data.to(device).view(-1,1,img_size,img_size,img_size).float()
                    o_data=o_data.to(device).view(-1,1,img_size,img_size,img_size).float()
                
                    t_label=t_label.to(device).view(-1,1,img_size,img_size,img_size).float()
                    o_label=o_label.to(device).view(-1,1,img_size,img_size,img_size).float()

                    outputs_L,flow_L,grid_L=model(o_data,t_data)
                    loss=loss_func(outputs_L[-1],o_data)
                    
                    total_sim_loss_test.append(loss.item())

                    pre_label=t_label
                    stage_dice=[]
                    for j in range(int(sqe_len)):
                        
                        cur_label = F.grid_sample(pre_label, grid_L[j],mode='nearest',align_corners=True)
                        dice_loss=compute_label_dice(o_label, cur_label)
                        stage_dice.append(dice_loss)
                        if sample_orig==False:
                            pre_label=cur_label
                        elif sample_orig==True:
                            pre_label=t_label
                      
                        #print (dice_loss)
                    total_dice_loss.append(stage_dice[-1])       
                            

                ave_sim_loss_test=torch.mean(torch.FloatTensor(total_sim_loss_test))
                std_sim_loss_test=torch.std(torch.FloatTensor(total_sim_loss_test))
                
                ave_dice_loss_test=torch.mean(torch.FloatTensor(total_dice_loss))
                std_dice_loss_test=torch.std(torch.FloatTensor(total_dice_loss))

                loss_info="Epoch[{}/{}], All Training loss: {:.4f}/{:.4f} , Sim Training loss: {:.4f}/{:.4f} , testing loss: {:.4f}/{:.4f}  ,  Dice loss: {:.4f}/{:.4f}".format(epoch+1,num_epochs,
                ave_loss_train,std_loss_train,
                ave_sim_loss_train,std_sim_loss_train,
                ave_sim_loss_test,std_sim_loss_test,
                ave_dice_loss_test,std_dice_loss_test)
                    
                                
                print (loss_info)
                append_log(loss_info,loss_log_path,modal_name)
                
                
                save_sample_any(epoch,"o",o_data,sample_o_path)
                save_nii_any(epoch,"o",o_data,sample_o_path)
                
                save_sample_any(epoch,"t",t_data,sample_t_path)
                save_nii_any(epoch,"t",t_data,sample_t_path)
                
                for k in range(int(sqe_len)):
                    idx=k+1
                    p_name="p_"+str(idx)
                    sample_p_path=modal_path+"/"+p_name
                    
                    save_sample_any(epoch,p_name,outputs_L[k],sample_p_path)
                    save_nii_any(epoch,p_name,outputs_L[k],sample_p_path)

                torch.save(model.state_dict(), os.path.join(model_save_path,modal_name+"_"+str(epoch)+".pth"))
    return
