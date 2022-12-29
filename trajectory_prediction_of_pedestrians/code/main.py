import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import loader
import os
import matplotlib.pyplot as plt 
import traceback
from environment import ApplicationProperties
from models import VanillaLSTMNet, GRUNet, SLSTM, OLSTM
import random
from torch.autograd import Variable
import pandas as pd
from tensorboardX import SummaryWriter
import math
import utils

def computeLSTMTrainingLoss(vanilla_lstm_net,pred_len=0):
    use_cuda = bool( applicationProperties.get_property_value( "lstm.train.cuda" ) )
    test_data_dir = os.path.join( applicationProperties.get_property_value( "lstm.train.testDir") )

    # retrieve dataloader
    dataset, dataloader = loader.data_loader(test_data_dir)

    # define parameters for training and testing loops
    criterion = nn.MSELoss() # MSE works best for difference between predicted and actual coordinate paths

    # initialize lists for capturing losses
    test_loss = []
    test_avgD_error=[]
    test_finalD_error=[]
    num_test_peds=0 #counter for number of pedestrians in the test dataset
    # now, test the model
    for i, batch in enumerate(dataloader):
        if(use_cuda): #when use_cuda has been mentioned in the argument
            test_observed_batch = batch[0].cuda()
            test_target_batch = batch[1].cuda()[0:pred_len]
        else: #when use_cuda has not been mentioned in the argument
            test_observed_batch = batch[0]
            test_target_batch = batch[1][0:pred_len]

        out = vanilla_lstm_net(test_observed_batch, pred_len=pred_len) # forward pass of lstm network for training
        print(out.shape)
        print(test_target_batch.shape)

        cur_test_loss = criterion(out, test_target_batch) # calculate MSE loss
        test_loss.append(cur_test_loss.item())
        out1=out
        target_batch1=test_target_batch  #making a copy of the tensors to convert them to array
        if(use_cuda): #to transfer the tensors back to CPU if cuda was used
            out1=out1.cpu()
            target_batch1=target_batch1.cpu()

        seq, peds, coords = test_target_batch.shape
        num_test_peds+=peds
        avgD_error=(np.sum(np.sqrt(np.square(out1[:,:,0].detach().numpy()-target_batch1[:,:,0].detach().numpy())+
            np.square(out1[:,:,1].detach().numpy()-target_batch1[:,:,1].detach().numpy()))))/(pred_len*peds)
        test_avgD_error.append(avgD_error)

        # final displacement error
        finalD_error=(np.sum(np.sqrt(np.square(out1[pred_len-1,:,0].detach().numpy()-target_batch1[pred_len-1,:,0].detach().numpy())+
            np.square(out1[pred_len-1,:,1].detach().numpy()-target_batch1[pred_len-1,:,1].detach().numpy()))))/peds
        test_finalD_error.append(finalD_error)
                
    avg_testloss = sum(test_loss)/len(test_loss)
    avg_testD_error=sum(test_avgD_error)/len(test_avgD_error)
    avg_testfinalD_error=sum(test_finalD_error)/len(test_finalD_error)
    print("============= Average test loss:", avg_testloss, "====================")


    return avg_testloss, avg_testD_error,avg_testfinalD_error,num_test_peds

def trainLSTMModel():
    '''define parameters for training and testing loops!'''

    num_epoch = applicationProperties.get_property_value( "lstm.train.epochs" )
    pred_len = applicationProperties.get_property_value( "lstm.train.predLen" )
    learning_rate = applicationProperties.get_property_value( "lstm.train.learningRate" )
    obs_len = applicationProperties.get_property_value( "lstm.train.obsLen" )
    use_cuda = bool( applicationProperties.get_property_value( "lstm.train.cuda" ) )
    data_dir = os.path.join( applicationProperties.get_property_value( "lstm.train.dataDir" ) )
    # retrieve dataloader
    _, dataloader = loader.data_loader( data_dir )

    ''' define the network, optimizer and criterion '''
    name = applicationProperties.get_property_value( "lstm.train.dataset" ) # to add to the name of files
    vanilla_lstm_net = VanillaLSTMNet()
    if(use_cuda):
        vanilla_lstm_net.cuda()

    criterion = nn.MSELoss() # MSE works best for difference between predicted and actual coordinate paths
    optimizer = optim.Adam(vanilla_lstm_net.parameters(), lr=learning_rate)

    # initialize lists for capturing losses/errors
    train_loss = []
    test_loss = []
    avg_train_loss = []
    avg_test_loss = []
    train_avgD_error=[]
    train_finalD_error=[]
    avg_train_avgD_error=[]
    avg_train_finalD_error=[]
    test_finalD_error=[]
    test_avgD_error=[]
    std_train_loss = []
    num_train_peds=[] #counter for number of pedestrians taken for each epoch


    '''training loop'''
    for i in range(num_epoch):
       
        print('======================= Epoch: {cur_epoch} / {total_epochs} =======================\n'.format(cur_epoch=i, total_epochs=num_epoch))
        
        def closure():
            train_peds=0 #counter for number of pedestrians taken for each epoch
            for i, batch in enumerate(dataloader):
                if(use_cuda):
                    train_batch = batch[0].cuda()
                    target_batch = batch[1].cuda()[0:pred_len]
                else:
                    train_batch = batch[0]
                    target_batch = batch[1][0:pred_len]

                seq, peds, coords = train_batch.shape # q is number of pedestrians 
                train_peds+=peds #keeping a count of the number of peds in the data
                out = vanilla_lstm_net(train_batch, pred_len=pred_len) # forward pass of lstm network for training
                optimizer.zero_grad() # zero out gradients
                cur_train_loss = criterion(out, target_batch) # calculate MSE loss
                print('Current training loss: {}'.format(cur_train_loss.item())) # print current training loss
                
                #calculating average deisplacement error
                out1=out
                target_batch1=target_batch  #making a copy of the tensors to convert them to array
                if(use_cuda):
                    out1=out1.cpu()
                    target_batch1=target_batch1.cpu()
                
                avgD_error=(np.sum(np.sqrt(np.square(out1[:,:,0].detach().numpy()-target_batch1[:,:,0].detach().numpy())+
                    np.square(out1[:,:,1].detach().numpy()-target_batch1[:,:,1].detach().numpy()))))/(pred_len*peds)
                train_avgD_error.append(avgD_error)

                #calculate final displacement error
                finalD_error=(np.sum(np.sqrt(np.square(out1[pred_len-1,:,0].detach().numpy()-target_batch1[pred_len-1,:,0].detach().numpy())+
                    np.square(out1[pred_len-1,:,1].detach().numpy()-target_batch1[pred_len-1,:,1].detach().numpy()))))/peds
                train_finalD_error.append(finalD_error)

                train_loss.append(cur_train_loss.item())
                cur_train_loss.backward() # backward prop
                optimizer.step() # step like a mini-batch (after all pedestrians)
            num_train_peds.append(train_peds)
            return cur_train_loss

        try:
          optimizer.step(closure) # update weights
        except Exception as e:
          traceback.print_exc()


        # save model at every epoch (uncomment) 
        # torch.save(lstm_net, './saved_models/lstm_model_v3.pt')
        # print("Saved lstm_net!")
        avg_train_loss.append(np.sum(train_loss)/len(train_loss))
        avg_train_avgD_error.append(np.sum(train_avgD_error)/len(train_avgD_error))
        avg_train_finalD_error.append(np.sum(train_finalD_error)/len(train_finalD_error))   
        std_train_loss.append(np.std(np.asarray(train_loss)))
        train_loss = [] # empty train loss

        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("average train loss: {}".format(avg_train_loss))
        print("average std loss: {}".format(std_train_loss))
        avgTestLoss,avgD_test,finalD_test,num_test_peds=computeLSTMTrainingLoss(vanilla_lstm_net,pred_len)
        avg_test_loss.append(avgTestLoss)
        test_finalD_error.append(finalD_test)
        test_avgD_error.append(avgD_test)
        print("test finalD error: ",finalD_test)
        print("test avgD error: ",avgD_test)


    '''after running through epochs, save your model and visualize.
       then, write your average losses and standard deviations of 
       losses to a text file for record keeping.'''

    save_path = os.path.join('./saved_models/', 'lstm_model_'+name+'_lr_' + str(learning_rate) + '_epoch_' + str(num_epoch) + '_predlen_' + str(pred_len) +'_obs'+str(obs_len)+ '.pt')
    # torch.save(lstm_net, './saved_models/lstm_model_lr001_ep20.pt')
    torch.save(vanilla_lstm_net, save_path)
    print("saved lstm_net! location: " + save_path)

    ''' visualize losses vs. epoch'''
    plt.figure() # new figure
    plt.title("Average train loss vs {} epochs".format(num_epoch))
    plt.plot(avg_train_loss,label='avg train_loss') 
    plt.plot(avg_test_loss,color='red',label='avg test_loss')
    plt.legend()
    plt.savefig("./saved_figs/" + "lstm_"+name+"_avgtrainloss_lr_"+ str(learning_rate) + '_epochs_' + str(num_epoch) + '_predlen_' + str(pred_len) +'_obs'+str(obs_len)+  '.png')
    # plt.show()
    # plt.show(block=True)
    
    plt.figure() # new figure
    plt.title("Average and final displacement error {} epochs".format(num_epoch))
    plt.plot(avg_train_finalD_error,label='train:final disp. error') 
    plt.plot(avg_train_avgD_error,color='red',label='train:avg disp. error')
    plt.plot(test_finalD_error,color='green',label='test:final disp. error')
    plt.plot(test_avgD_error,color='black',label='test:avg disp. error')
    plt.ylim((0,10))
    plt.legend()
    # plt.show()
    plt.savefig("./saved_figs/" + "lstm_"+name+"_avg_final_displacement_lr_"+ str(learning_rate) + '_epochs_' + str(num_epoch) + '_predlen_' + str(pred_len) +'_obs'+str(obs_len)+  '.png')

    plt.figure()
    plt.title("Std of train loss vs epoch{} epochs".format(num_epoch))
    plt.plot(std_train_loss)
    plt.savefig("./saved_figs/" + "lstm_"+name+"_stdtrainloss_lr_"+ str(learning_rate) + '_epochs_' + str(num_epoch) + '_predlen_' + str(pred_len) +'_obs'+str(obs_len)+'.png')
    # plt.show(block=True)
    print("saved images for avg training losses! location: " + "./saved_figs")

    # save results to text file
    txtfilename = os.path.join("./txtfiles/", "lstm_"+name+"_avgtrainlosses_lr_"+ str(learning_rate) + '_epochs_' + str(num_epoch) + '_predlen_' + str(pred_len) +'_obs'+str(obs_len)+ ".txt")
    os.makedirs(os.path.dirname("./txtfiles/"), exist_ok=True) # make directory if it doesn't exist
    with open(txtfilename, "w") as f:
        f.write("Number of pedestrians in the traning data: {}\n".format(num_train_peds[-1]))
        f.write("Number of pedestrians in the test dataset: {}\n".format(num_test_peds))
        f.write("\n==============Average train loss vs. epoch:===============\n")
        f.write(str(avg_train_loss))
        f.write("\nepochs: " + str(num_epoch))
        f.write("\n==============Std train loss vs. epoch:===================\n")
        f.write(str(std_train_loss))
        f.write("\n==============Avg test loss vs. epoch:===================\n")
        f.write(str(avg_test_loss))
        f.write("\n==============Avg train displacement error:===================\n")
        f.write(str(avg_train_avgD_error))
        f.write("\n==============Final train displacement error:===================\n")
        f.write(str(avg_train_finalD_error))
        f.write("\n==============Avg test displacement error:===================\n")
        f.write(str(test_avgD_error))
        f.write("\n==============Final test displacement error:===================\n")
        f.write(str(test_finalD_error))
    print("saved average and std of training losses to text file in: ./txtfiles")

    # saving all data files with different prediction lengths and observed lengths for each dataset    
    txtfilename2 = os.path.join("./txtfiles/", "RESULTS_LSTM2"+"_diff_ObsPred_len_lr_"+ str(learning_rate) + '_numepochs_' + str(num_epoch)+ ".txt")
    os.makedirs(os.path.dirname("./txtfiles/"), exist_ok=True) # make directory if it doesn't exist
    with open(txtfilename2,"a+") as g: #opening the file in the append mode
        if(pred_len==2):
            g.write("Dataset: "+name+" ;Number of epochs: {}".format(num_epoch)+"\n")
            g.write("obs_len"+"\t"+"pred_len"+"\t"+"avg_train_loss"+"\t"+"avg_test_loss"+"\t"+"std_train_loss"+"\t"
            	+"avg_train_dispacement"+"\t"+"final_train_displacement"+"\t"+"avg_test_displacement"+"\t"
            	+"final_test_displacement"+"\t"+"Num_train_peds"+"\t"+"Num_Test_peds"+"\n")
        # outputing the current observed length
        g.write(str(obs_len)+"\t")
        # outputing the current prediction length
        g.write(str(pred_len)+"\t")
        #the avg_train_loss after total epochs
        g.write(str(avg_train_loss[-1])+"\t")
        # the avg_test_loss after total epochs
        g.write(str(avg_test_loss[-1])+"\t")
        # the standard deviation of train loss
        g.write(str(std_train_loss[-1])+"\t")
        # the avg train dispacement error
        g.write(str(avg_train_avgD_error[-1])+"\t")
        # the train final displacement error
        g.write(str(avg_train_finalD_error[-1])+"\t")
        # the test avg displacement error
        g.write(str(test_avgD_error[-1])+"\t")
        # the test final displacement error
        g.write(str(test_finalD_error[-1])+"\t")
        # the number of pedestrians in the traininig dataset
        g.write(str(num_train_peds[-1])+"\t")
        # Number of pedestrian sin the training dataset
        g.write(str(num_test_peds)+"\n")
    print("saved all the results to the text file for observed length: {}".format(obs_len))

def testLSTMModel():
    vanilla_lstm_net = torch.load( applicationProperties.get_property_value( "lstm.test.model" ) )
    vanilla_lstm_net.eval()

    test_data_dir = os.path.join( applicationProperties.get_property_value( "lstm.test.testDir") )

    # retrieve dataloader
    _, dataloader = loader.data_loader(test_data_dir)

    # define parameters for training and testing loops
    pred_len = applicationProperties.get_property_value( "lstm.test.predLen" )
    criterion = nn.MSELoss() # MSE works best for difference between predicted and actual coordinate paths

    # initialize lists for capturing losses
    test_loss = []

    # now, test the model
    for _, batch in enumerate(dataloader):
      test_observed_batch = batch[0]
      test_target_batch = batch[1][0:pred_len]
      out = vanilla_lstm_net(test_observed_batch, pred_len=pred_len) # forward pass of lstm network for training
      print("out's shape:", out.shape)
      cur_test_loss = criterion(out, test_target_batch) # calculate MSE loss
      print('Current test loss: {}'.format(cur_test_loss.item())) # print current test loss
      test_loss.append(cur_test_loss.item())
    avg_testloss = sum(test_loss)/len(test_loss)
    print("========== Average test loss:", avg_testloss, "==========")

def visualizeLSTM():
    cur_dataset = applicationProperties.get_property_value( "lstm.visualize.dataset" )
    pred_len = applicationProperties.get_property_value( "lstm.visualize.predLen" )
    test_data_dir = os.path.join( applicationProperties.get_property_value( "lstm.visualize.testDir") )
    
    lstm_net = torch.load( applicationProperties.get_property_value( "lstm.visualize.model" ) )
    lstm_net.eval()

    _, dataloader = loader.data_loader(test_data_dir)

    plt.xlabel("X coordinates of pedestrians")
    plt.ylabel("Y coordinates of pedestrians")
    # now, test the model
    for _, batch in enumerate(dataloader):
        test_observed_batch = batch[0]
        test_target_batch = batch[1]
        out = lstm_net(test_observed_batch, pred_len=pred_len) # forward pass of lstm network for training
        s,_,_=out.shape
        out1=out.detach().numpy()
        target1=test_target_batch.detach().numpy()
        observed1=test_observed_batch.detach().numpy()
        print("observed 1 shape:",observed1.shape)
        print("target1 shape:", target1.shape)
        print("out 1 shape", out1.shape)
        out2=np.vstack((observed1,out1))
        target2=np.vstack((observed1,target1))
        print("out2 shape",out2.shape)
        for t in range(12):
            plt.plot(observed1[:,t,0],observed1[:,t,1],color='b',marker='o',linewidth=5,markersize=12)
            plt.plot(target2[s-1:s+pred_len,t,0],target2[s-1:s+pred_len,t,1],color='red',marker='o',linewidth=5,markersize=12)
            plt.plot(out2[s-1:s+pred_len,t,0],out2[s-1:s+pred_len,t,1],color='g',marker='o',linewidth=5,markersize=12)
        plt.legend(["Observed","Ground Truth","Predicted"])
        plt.show(block=True)

def computeGRUTrainingLoss(gru_net, pred_len=0):
    use_cuda = bool( applicationProperties.get_property_value( "gru.train.cuda" ) )
    test_data_dir = os.path.join( applicationProperties.get_property_value( "gru.train.testDir") )

    dataset, dataloader = loader.data_loader( test_data_dir)

    # define parameters for training and testing loops
    criterion = nn.MSELoss() # MSE works best for difference between predicted and actual coordinate paths
    num_test_peds=0 #counter for the number of pedestrians in the test data
    # initialize lists for capturing losses
    test_loss = []
    test_avgD_error=[]
    test_finalD_error=[]
   
    # now, test the model
    for i, batch in enumerate(dataloader):
        if(use_cuda):
            test_observed_batch = batch[0].cuda()
            test_target_batch = batch[1].cuda()[0:pred_len]
        else:
            test_observed_batch = batch[0]
            test_target_batch = batch[1][0:pred_len]

        out = gru_net(test_observed_batch, pred_len=pred_len) # forward pass of gru network for training
        cur_test_loss = criterion(out, test_target_batch) # calculate MSE loss
        test_loss.append(cur_test_loss.item())
        out1=out
        target_batch1=test_target_batch  #making a copy of the tensors to convert them to array
        if(use_cuda):
            # out1=out1.cpu()
            out1=out1.cuda()
            target_batch1=target_batch1.cuda()
            # target_batch1=target_batch1.cpu()
        
        seq, peds, coords = test_target_batch.shape
        num_test_peds+=peds
        avgD_error=(np.sum(np.sqrt(np.square(out1[:,:,0].detach().numpy()-target_batch1[:,:,0].detach().numpy())+
            np.square(out1[:,:,1].detach().numpy()-target_batch1[:,:,1].detach().numpy()))))/(pred_len*peds)
        test_avgD_error.append(avgD_error)

        # final displacement error
        finalD_error=(np.sum(np.sqrt(np.square(out1[pred_len-1,:,0].detach().numpy()-target_batch1[pred_len-1,:,0].detach().numpy())+
            np.square(out1[pred_len-1,:,1].detach().numpy()-target_batch1[pred_len-1,:,1].detach().numpy()))))/peds
        test_finalD_error.append(finalD_error)
                
    avg_testloss = sum(test_loss)/len(test_loss)
    avg_testD_error=sum(test_avgD_error)/len(test_avgD_error)
    avg_testfinalD_error=sum(test_finalD_error)/len(test_finalD_error)
    print("============= Average test loss:", avg_testloss, "====================")


    return avg_testloss, avg_testD_error,avg_testfinalD_error,num_test_peds

def trainGRUModel():
    num_epoch = applicationProperties.get_property_value( "gru.train.epochs" )
    pred_len = applicationProperties.get_property_value( "gru.train.predLen" )
    learning_rate = applicationProperties.get_property_value( "gru.train.learningRate" )
    obs_len = applicationProperties.get_property_value( "gru.train.obsLen" )
    use_cuda = bool( applicationProperties.get_property_value( "gru.train.cuda" ) )
    data_dir = os.path.join( applicationProperties.get_property_value( "gru.train.dataDir" ) )
    # retrieve dataloader
    _, dataloader = loader.data_loader( data_dir )

    name = applicationProperties.get_property_value( "gru.train.dataset" ) # to add to the name of files

    gru_net = GRUNet()
    if(use_cuda):
        gru_net.cuda()
    
    criterion = nn.MSELoss() # MSE works best for difference between predicted and actual coordinate paths
    optimizer = optim.Adam(gru_net.parameters(), lr=learning_rate)

    # initialize lists for capturing losses/errors
    train_loss = []
    test_loss = []
    avg_train_loss = []
    avg_test_loss = []
    train_avgD_error=[]
    train_finalD_error=[]
    avg_train_avgD_error=[]
    avg_train_finalD_error=[]
    test_finalD_error=[]
    test_avgD_error=[]
    std_train_loss = []
    std_test_loss = []
    num_train_peds=[] #counter for the number of pedestrians in the training data

    '''training loop'''
    for i in range(num_epoch):
        print('======================= Epoch: {cur_epoch} / {total_epochs} =======================\n'.format(cur_epoch=i, total_epochs=num_epoch))
        def closure():
            train_peds=0 #counter for the number of pedestrians in the training data
            for i, batch in enumerate(dataloader):
                if(use_cuda):
                    train_batch = batch[0].cuda()
                    target_batch = batch[1].cuda()[0: pred_len]
                else:
                    train_batch = batch[0]
                    target_batch = batch[1][0: pred_len]
                # print("train_batch's shape", train_batch.shape)
                # print("target_batch's shape", target_batch.shape)
                seq, peds, coords = train_batch.shape # q is number of pedestrians
                train_peds+=peds 
                out = gru_net(train_batch, pred_len=pred_len) # forward pass of gru network for training
                # print("out's shape:", out.shape)
                optimizer.zero_grad() # zero out gradients
                cur_train_loss = criterion(out, target_batch) # calculate MSE loss
                # print('Current training loss: {}'.format(cur_train_loss.item())) # print current training loss
                print('Current training loss: {}'.format(cur_train_loss.item())) # print current training loss
                
                #calculating average deisplacement error
                out1=out
                target_batch1=target_batch  #making a copy of the tensors to convert them to array
                if(use_cuda):
                    # out1=out1.cpu()
                    out1=out1.cuda()
                    target_batch1=target_batch1.cuda()
                    # target_batch1=target_batch1.cpu()
                avgD_error=(np.sum(np.sqrt(np.square(out1[:,:,0].detach().numpy()-target_batch1[:,:,0].detach().numpy())+
                    np.square(out1[:,:,1].detach().numpy()-target_batch1[:,:,1].detach().numpy()))))/(pred_len*peds)
                train_avgD_error.append(avgD_error)

                #calculate final displacement error
                finalD_error=(np.sum(np.sqrt(np.square(out1[pred_len-1,:,0].detach().numpy()-target_batch1[pred_len-1,:,0].detach().numpy())+
                    np.square(out1[pred_len-1,:,1].detach().numpy()-target_batch1[pred_len-1,:,1].detach().numpy()))))/peds
                train_finalD_error.append(finalD_error)

                train_loss.append(cur_train_loss.item())
                cur_train_loss.backward() # backward prop
                optimizer.step() # step like a mini-batch (after all pedestrians)
            num_train_peds.append(train_peds)
            return cur_train_loss
        optimizer.step(closure) # update weights

        # save model at every epoch (uncomment) 
        # torch.save(gru_net, './saved_models/gru_model_v3.pt')
        # print("Saved gru_net!")
        avg_train_loss.append(np.sum(train_loss)/len(train_loss))
        avg_train_avgD_error.append(np.sum(train_avgD_error)/len(train_avgD_error))
        avg_train_finalD_error.append(np.sum(train_finalD_error)/len(train_finalD_error))   
        std_train_loss.append(np.std(np.asarray(train_loss)))
        train_loss = [] # empty train loss

        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("average train loss: {}".format(avg_train_loss))
        print("average std loss: {}".format(std_train_loss))
        avgTestLoss,avgD_test,finalD_test,num_test_peds=computeGRUTrainingLoss(gru_net,pred_len)
        avg_test_loss.append(avgTestLoss)
        test_finalD_error.append(finalD_test)
        test_avgD_error.append(avgD_test)
        print("test finalD error: ",finalD_test)
        print("test avgD error: ",avgD_test)
        print("Number of pedestrians: {}".format(num_train_peds))
        #avg_test_loss.append(test(gru_net,pred_len)) ##calliing test function to return avg test loss at each epoch


    '''after running through epochs, save your model and visualize.
       then, write your average losses and standard deviations of 
       losses to a text file for record keeping.'''

    save_path = os.path.join('./saved_models/', 'gru_model_'+name+'_lr_' + str(learning_rate) + '_epoch_' + str(num_epoch) + '_predlen_' + str(pred_len) +'_obs'+str(obs_len)+ '.pt')
    # torch.save(gru_net, './saved_models/gru_model_lr001_ep20.pt')
    torch.save(gru_net, save_path)
    print("saved gru_net! location: " + save_path)

    ''' visualize losses vs. epoch'''
    plt.figure() # new figure
    plt.title("Average train loss vs {} epochs".format(num_epoch))
    plt.plot(avg_train_loss,label='avg train_loss') 
    plt.plot(avg_test_loss,color='red',label='avg test_loss')
    plt.legend()
    plt.savefig("./saved_figs/" + "gru_"+name+"_avgtrainloss_lr_"+ str(learning_rate) + '_epochs_' + str(num_epoch) + '_predlen_' + str(pred_len) +'_obs'+str(obs_len)+  '.png')
    # plt.show()
    # plt.show(block=True)
    
    plt.figure() # new figure
    plt.title("Average and final displacement error {} epochs".format(num_epoch))
    plt.plot(avg_train_finalD_error,label='train:final disp. error') 
    plt.plot(avg_train_avgD_error,color='red',label='train:avg disp. error')
    plt.plot(test_finalD_error,color='green',label='test:final disp. error')
    plt.plot(test_avgD_error,color='black',label='test:avg disp. error')
    plt.ylim((0,10))
    plt.legend()
    # plt.show()
    plt.savefig("./saved_figs/" + "gru_"+name+"_avg_final_displacement_lr_"+ str(learning_rate) + '_epochs_' + str(num_epoch) + '_predlen_' + str(pred_len) +'_obs'+str(obs_len)+  '.png')

    plt.figure()
    plt.title("Std of train loss vs epoch{} epochs".format(num_epoch))
    plt.plot(std_train_loss)
    plt.savefig("./saved_figs/" + "gru_"+name+"_stdtrainloss_lr_"+ str(learning_rate) + '_epochs_' + str(num_epoch) + '_predlen_' + str(pred_len) +'_obs'+str(obs_len)+'.png')
    # plt.show(block=True)
    print("saved images for avg training losses! location: " + "./saved_figs")

    # save results to text file
    txtfilename = os.path.join("./txtfiles/", "gru_"+name+"_avgtrainlosses_lr_"+ str(learning_rate) + '_epochs_' + str(num_epoch) + '_predlen_' + str(pred_len) +'_obs'+str(obs_len)+ ".txt")
    os.makedirs(os.path.dirname("./txtfiles/"), exist_ok=True) # make directory if it doesn't exist
    with open(txtfilename, "w") as f:
        f.write("Number of pedestrians in the training data: {}\n".format(num_train_peds[-1]))    
        f.write("Number of pedestrians in the testing data: {}\n".format(num_test_peds))  
        f.write("\n==============Average train loss vs. epoch:===============\n")
        f.write(str(avg_train_loss))
        f.write("\nepochs: " + str(num_epoch))
        f.write("\n==============Std train loss vs. epoch:===================\n")
        f.write(str(std_train_loss))
        f.write("\n==============Avg test loss vs. epoch:===================\n")
        f.write(str(avg_test_loss))
        f.write("\n==============Avg train displacement error:===================\n")
        f.write(str(avg_train_avgD_error))
        f.write("\n==============Final train displacement error:===================\n")
        f.write(str(avg_train_finalD_error))
        f.write("\n==============Avg test displacement error:===================\n")
        f.write(str(test_avgD_error))
        f.write("\n==============Final test displacement error:===================\n")
        f.write(str(test_finalD_error))
    print("saved average and std of training losses to text file in: ./txtfiles")
    
    txtfilename2 = os.path.join("./txtfiles/", "GRU_RESULTS"+name+"_diff_obs_pred_len_lr_"+ str(learning_rate) + '_epochs_' + str(num_epoch)+ ".txt")
    os.makedirs(os.path.dirname("./txtfiles/"), exist_ok=True) # make directory if it doesn't exist
    with open(txtfilename2,"a+") as g: #opening the file in the append mode
        if(pred_len==2):
            g.write("Dataset: "+name+" ;Number of epochs: {}".format(num_epoch)+"\n")
            g.write("obs_len"+"\t"+"pred_len"+"\t"+"avg_train_loss"+"\t"+"avg_test_loss"+"\t"+"std_train_loss"+"\t"
                +"avg_train_dispacement"+"\t"+"final_train_displacement"+"\t"+"avg_test_displacement"+"\t"+
                "final_test_displacement"+"Num_Train_peds"+"\t"+"Num_Test_Peds"+"\n")
        # outputing the current observed length
        g.write(str(obs_len)+"\t")
        # outputing the current prediction length
        g.write(str(pred_len)+"\t")
        #the avg_train_loss after total epochs
        g.write(str(avg_train_loss[-1])+"\t")
        # the avg_test_loss after total epochs
        g.write(str(avg_test_loss[-1])+"\t")
        # the standard deviation of train loss
        g.write(str(std_train_loss[-1])+"\t")
        # the avg train dispacement error
        g.write(str(avg_train_avgD_error[-1])+"\t")
        # the train final displacement error
        g.write(str(avg_train_finalD_error[-1])+"\t")
        # the test avg displacement error
        g.write(str(test_avgD_error[-1])+"\t")
        # the test final displacement error
        g.write(str(test_finalD_error[-1])+"\t")
        # the number of pedestrians in the traininig dataset
        g.write(str(num_train_peds[-1])+"\t")
        # Number of pedestrian sin the training dataset
        g.write(str(num_test_peds)+"\n")
    print("saved all the results to the text file for observed length: {}".format(obs_len))

def visualizeGRU():
    cur_dataset = applicationProperties.get_property_value( "gru.visualize.dataset" )
    pred_len = applicationProperties.get_property_value( "gru.visualize.predLen" )
    test_data_dir = os.path.join( applicationProperties.get_property_value( "gru.visualize.testDir") )
    use_cuda = bool( applicationProperties.get_property_value( "gru.visualize.cuda" ) )
    gru_net = torch.load( applicationProperties.get_property_value( "gru.visualize.model" ) )
    if (use_cuda):
        gru_net = gru_net.cuda()
    gru_net.eval()

    _, dataloader = loader.data_loader(test_data_dir)
    
    criterion = nn.MSELoss()

    test_loss = []
    test_avgD_error=[]
    test_finalD_error=[]
    # plt.figure(figsize=(32,20))
    plt.xlabel("X coordinates of pedestrians")
    plt.ylabel("Y coordinates of pedestrians")
    # now, test the model
    for i, batch in enumerate(dataloader):
        test_observed_batch = batch[0].cuda()
        test_target_batch = (batch[1].cuda())[0:pred_len]
        out = gru_net(test_observed_batch, pred_len=pred_len) # forward pass of lstm network for training
        cur_test_loss = criterion(out, test_target_batch) # calculate MSE loss
        test_loss.append(cur_test_loss.item())

        s,peds,c=out.shape
        out1=out.cpu().detach().numpy()
        target1=test_target_batch.cpu().detach().numpy()
        observed1=test_observed_batch.cpu().detach().numpy()
        print("observed 1 shape:",observed1.shape)
        print("target1 shape:", target1.shape)
        print("out 1 shape", out1.shape)
        out2=np.vstack((observed1,out1))
        target2=np.vstack((observed1,target1))
        print("out2 shape",out2.shape)
        avgD_error=(np.sum(np.sqrt(np.square(out1[:,:,0]-target1[:,:,0])+
            np.square(out1[:,:,1]-target1[:,:,1]))))/(pred_len*peds)
        test_avgD_error.append(avgD_error)

        # final displacement error
        finalD_error=(np.sum(np.sqrt(np.square(out1[pred_len-1,:,0]-target1[pred_len-1,:,0])+
            np.square(out1[pred_len-1,:,1]-target1[pred_len-1,:,1]))))/peds
        test_finalD_error.append(finalD_error)
        for t in range(8):
            plt.plot(observed1[:,t,0],observed1[:,t,1],color='b',marker='o',linewidth=5,markersize=12)
        for t in range(8):    
            plt.plot(target2[s-1:s+pred_len,t,0],target2[s-1:s+pred_len,t,1],color='red',marker='o',linewidth=5,markersize=12)
            plt.plot(out2[s-1:s+pred_len,t,0],out2[s-1:s+pred_len,t,1],color='g',marker='o',linewidth=5,markersize=12)
        plt.legend(["Observed","Ground Truth","Predicted"])
        plt.show(block=True)
                
    avg_testloss = sum(test_loss)/len(test_loss)
    avg_testD_error=sum(test_avgD_error)/len(test_avgD_error)
    avg_testfinalD_error=sum(test_finalD_error)/len(test_finalD_error)

        # out1=out
        # target_batch1=test_target_batch  #making a copy of the tensors to convert them to array
        # seq, peds, coords = test_target_batch.shape

    return avg_testloss,avg_testD_error,avg_testfinalD_error

def testSLSTMModel(test_data, save_plot, epoch, it, save_dir, num=12, comment='', plot_loss=False):
    
    batch_size = 1
    nb_iter = applicationProperties.get_property_value( "slstm.model.nb_iter" )
    embedded_input = applicationProperties.get_property_value( "slstm.model.embedded_input" )
    hidden_size = applicationProperties.get_property_value( "slstm.model.hidden_size" )
    max_dist_scaled = 84
    use_speeds = False
    # Instanciate model and data loader
    args = {'embedded_input': embedded_input, 'hidden_size': hidden_size,
            'grid_size': applicationProperties.get_property_value( "slstm.model.grid_size" ), 'max_dist': max_dist_scaled,
            'embedding_occupancy_map': applicationProperties.get_property_value( "slstm.model.embedding_occupancy_map_size" ), 
            'use_speeds': use_speeds,
            'trained_model': applicationProperties.get_property_value( "slstm.model.trained_model" )}
    
    net = SLSTM(args)
    net = utils.en_cuda(net)

    # Variables
    saved_for_plots = []
    obs = 8
    loss_display_test, rmse_display, l2_display, final_l2_display = 0, 0, 0, 0

    # Sample plots to save
    saved_plots = random.sample(range(len(test_data)), 3)
    saved_plots = dict([(int(math.floor(x / test_data.batch_size)), x %
                         test_data.batch_size) for x in saved_plots])

    for idx, batch in enumerate(test_data):

     # Count number of samples that we iterated on
        batch_for_network = None
        if use_speeds:
            batch_for_network = batch[1][0]
        else:
            batch_for_network = batch[0][0]

        batch_for_network = Variable(batch_for_network.squeeze(0))
        grids = batch[4][0]
        # Forward through network
        first_pos = torch.cat(batch[0], 0)[:, 7, :][:, [2, 3]]
        results, pts = net.forward(
            batch_for_network[:obs, :], grids, batch[2][0], first_pos)

        # Compute loss
        batch_for_network = batch_for_network.unsqueeze(1)
        loss = utils.get_lossfunc(results[:, :, 0], results[:, :, 1], results[:, :, 2], results[:, :, 3],
                                  results[:, :, 4], batch_for_network[obs:, :, 2], batch_for_network[obs:, :, 3])
        pts_predicted = pts

        # Compute accuracies
        tmp_true_pos = torch.cat(batch[0], 0).transpose(0, 1)
        acc = utils.get_avg_displacement(
            tmp_true_pos[obs:, :, :][:, :, [2, 3]], pts_predicted)
        acc_l2, acc_final = utils.get_accuracy(
            tmp_true_pos[obs:, :, :][:, :, [2, 3]], pts_predicted)

        if len(saved_for_plots) < 50 and random.choice([True, False]):
            saved_for_plots.append((batch, pts_predicted))

        loss_display_test += loss.data[0]
        acc = (torch.sum(acc))
        acc_l2 = torch.sum(acc_l2)
        rmse_display += acc
        l2_display += acc_l2
        final_l2_display += torch.sum(acc_final)

        if (idx in saved_plots) and save_plot:
            plot_true, plot_pred = batch, pts_predicted
            x = saved_plots[idx]
            utils.plot_trajectory(batch[0][x].squeeze(0), plot_pred[:, x, :], batch[2][
                                  x], 5, "{}social_lstm_it_{}_i_{}_b_{}_test_{}".format(save_dir[0], epoch, idx, x, comment))

    if save_plot:
        utils.save_checkpoint({
            'true_pred': saved_for_plots,
        }, save_dir[1] + 'test_data_plot_social_lstm_{}.pth.tar'.format(comment))

    print('[Loss_Test_{}] {}'.format(
        comment, loss_display_test / len(test_data)))
    print('[Accuracy_mse_Test_{}] {}'.format(comment,
                                             rmse_display / len(test_data)))
    print('[Accuracy_l2_Test_{}] {}'.format(
        comment, l2_display / len(test_data)))
    print('[Accuracy_final_Test_{}] {}'.format(comment,
                                               final_l2_display / len(test_data)))
    if plot_loss:
        print('Loss Test {}'.format(comment),
                          (loss_display_test / len(test_data)), it/64)
        print('acc_l2 Test {}'.format(comment),
                          (l2_display / len(test_data)), it/64)
    print('-------------------------------------------------------------')

def testOLSTMModel(test_data, save_plot, epoch, it, save_dir, num=12, comment='', plot_loss=False):
    
    batch_size = 1
    nb_iter = applicationProperties.get_property_value( "slstm.model.nb_iter" )
    embedded_input = applicationProperties.get_property_value( "slstm.model.embedded_input" )
    hidden_size = applicationProperties.get_property_value( "slstm.model.hidden_size" )
    max_dist_scaled = 84
    use_speeds = False
    # Instanciate model and data loader
    args = {'embedded_input': embedded_input, 'hidden_size': hidden_size,
            'grid_size': applicationProperties.get_property_value( "slstm.model.grid_size" ), 'max_dist': max_dist_scaled,
            'embedding_occupancy_map': applicationProperties.get_property_value( "slstm.model.embedding_occupancy_map_size" ), 
            'use_speeds': use_speeds}
    
    net = OLSTM(args)
    net = utils.en_cuda(net)

    it = 0
    min_epoch = 0
    optimizer = torch.optim.RMSprop(net.parameters(), lr=applicationProperties.get_property_value( "slstm.model.lr" ))

    # Variables
    saved_for_plots = []
    obs = 8
    loss_display_test, rmse_display, l2_display, final_l2_display = 0, 0, 0, 0

    # Sample plots to save
    saved_plots = random.sample(range(len(test_data)), 3)
    saved_plots = dict([(int(math.floor(x / test_data.batch_size)), x %
                         test_data.batch_size) for x in saved_plots])

    for idx, batch in enumerate(test_data):

     # Count number of samples that we iterated on
        batch_for_network = None
        if use_speeds:
            batch_for_network = batch[1][0]
        else:
            batch_for_network = batch[0][0]

        batch_for_network = Variable(batch_for_network.squeeze(0))
        grids = batch[4][0]
        # Forward through network
        first_pos = torch.cat(batch[0], 0)[:, 7, :][:, [2, 3]]
        results, pts = net.forward(
            batch_for_network[:obs, :], grids, batch[2][0], first_pos)

        # loss
        batch_for_network = batch_for_network.unsqueeze(1)
        loss = utils.get_lossfunc(results[:, :, 0], results[:, :, 1], results[:, :, 2], results[:, :, 3],
                                  results[:, :, 4], batch_for_network[obs:, :, 2], batch_for_network[obs:, :, 3])
        pts_predicted = pts

        # accuracies
        tmp_true_pos = torch.cat(batch[0], 0).transpose(0, 1)
        acc = utils.get_avg_displacement(tmp_true_pos[obs:, :, :][:, :, [2, 3]], pts_predicted)
        acc_l2, acc_final = utils.get_accuracy(tmp_true_pos[obs:, :, :][:, :, [2, 3]], pts_predicted)

        if len(saved_for_plots) < 50 and random.choice([True, False]):
            saved_for_plots.append((batch, pts_predicted))

        loss_display_test += loss.data[0]
        acc = (torch.sum(acc))
        acc_l2 = torch.sum(acc_l2)
        rmse_display += acc
        l2_display += acc_l2
        final_l2_display += torch.sum(acc_final)

        if (idx in saved_plots) and save_plot:
            plot_true, plot_pred = batch, pts_predicted
            x = saved_plots[idx]

    print('[Loss_Test_{}] {}'.format(comment, loss_display_test / len(test_data)))
    print('[Accuracy_mse_Test_{}] {}'.format(comment, rmse_display / len(test_data)))
    print('[Accuracy_l2_Test_{}] {}'.format(comment, l2_display / len(test_data)))
    print('[Accuracy_final_Test_{}] {}'.format(comment, final_l2_display / len(test_data)))
    print('Loss Test {}'.format(comment), (loss_display_test / len(test_data)), it/64)
    print('acc_l2 Test {}'.format(comment), (l2_display / len(test_data)), it/64)

if __name__ == "__main__":
    applicationProperties = ApplicationProperties("application.yml")
    applicationProperties.initializeProperties()

    # Training LSTM Model
    trainLSTMModel()

    # Testing LSTM Model
    testLSTMModel()

    # Visualizing LSTM Model
    visualizeLSTM() 