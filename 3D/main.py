from train import *

if __name__ == "__main__":

    
    img_size=96
    num_stage=10
    train_set_name="LPBA40_train_sub.npy"
    test_set_name="LPBA40_test_sub.npy"
    batch_size=1
    num_epochs=1000
    model_name=ABN_3D(img_size,num_stage)   #ABN_3D or ABN_L_3D
    loss_name="NCC"
    penalty="l2"
    smooth_name=second_Grad(penalty)
    learning_rate=0.0001
    lamda=10
    save_every_epoch=10
    sample_orig=True
    
    
    train(img_size,
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
          sample_orig)