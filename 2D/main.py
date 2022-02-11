from train import *


if __name__ == "__main__":
    
    train_set_name="2D_face_train.npy"
    test_set_name="2D_face_test.npy"
    penalty="l2"
    lamda=10
    loss_name="MSE" 
    num_stage=10
    num_lines=15
    grid_sample_orig=True
    batch_size=16
    img_size=64
    num_epochs=5000
    model_name=ABN_L_2D(img_size,num_stage)    # ABN_2D or ABN_L_2D
    smooth_name=second_Grad(penalty)
    learning_rate=0.0001
    save_every_epoch=20
    num_sample=16

    train_2d(img_size,
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
              grid_sample_orig)