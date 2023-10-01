import os 
import pandas as pd 
import numpy as np
from tqdm import tqdm 
import pickle 
from collections import Counter
#read each file and extract the labels

def read_txt_file(txt_file_path):

    label_list=[]
    print('Loading the file:%s' %(txt_file_path))
    with open(txt_file_path, 'r') as f:
        lines = f.readlines()
        #lines = [x.strip() for x in lines]
        

    for line in tqdm(lines):
        label_c=line.strip().split(' ')[1:]
        label_c=[int(l) for l in label_c]
        label_list=label_list+label_c

    return label_list


train_file="../split_files/train_multi_label_thresh_0_4_0_1_150_labels.txt"
val_file="../split_files/val_multi_label_thresh_0_4_0_1_150_labels.txt"
test_file="../split_files/test_multi_label_thresh_0_4_0_1_150_labels.txt"
label_map_file="../split_files/label_2_ind_multi_label_thresh_0_4_0_1_150_label_map.pkl"


train_labels=read_txt_file(train_file)
val_labels=read_txt_file(val_file)
test_labels=read_txt_file(test_file)

total_labels=train_labels+val_labels+test_labels

with open(label_map_file, 'rb') as f:
    label_map = pickle.load(f)

#obtain the reverse map 
reverse_label_map={v:k for k,v in label_map.items()}

#compute the distribution of the labels
label_names=[reverse_label_map[l] for l in total_labels]

label_occurence=Counter(label_names).most_common(150)

#save counter dict as dataframe
df=pd.DataFrame(label_occurence,columns=['label','count'])

df.to_csv('../split_files/label_distribution_multi_label_thresh_0_4_0_1_150.csv',index=False)









