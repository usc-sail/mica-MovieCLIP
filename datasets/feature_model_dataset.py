from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os 
import torch 
import torchvision.transforms as transforms
import pandas as pd 
import numpy as np 

#dataset declcations for Multi label classification using LSTM 
class LSTM_Multi_Label_Dataset(Dataset):
    def __init__(self,folder,feat_lines,max_len=20,num_classes=150):
        
        self.folder=folder
        self.max_len=max_len 
        self.feat_lines=feat_lines
        self.num_classes=num_classes
        
    def __len__(self):
        return(len(self.feat_lines))

    def __getitem__(self,idx):

        c_list=self.feat_lines[idx].split(" ")
        filename_c=os.path.join(self.folder,c_list[0].split("/")[-1][:-4]+".npy")
        label_list=c_list[1:]
        label_vect=np.zeros((self.num_classes))
        for l in label_list:
            label_vect[int(l)]=1

        label=torch.LongTensor(label_vect)    

        feat_data=np.load(filename_c)
        if(feat_data.shape[0]>=self.max_len):
            len_sample=self.max_len
        else:
            len_sample=feat_data.shape[0]

        feat_data=self.pad_data(feat_data)
        feat_data=torch.Tensor(feat_data)

        return(feat_data,label,len_sample)

    def pad_data(self,feat_data):
        padded=np.zeros((self.max_len,feat_data.shape[1]))
        if(feat_data.shape[0]>self.max_len):
            padded=feat_data[:self.max_len,:]
        else:
            padded[:feat_data.shape[0],:]=feat_data
        return(padded)