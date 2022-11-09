import os 
import pandas as pd 

#path to txt file containing train, val and test 
train_file="../split_files/train_multi_label_thresh_0_4_0_1_150_labels.txt"
val_file="../split_files/val_multi_label_thresh_0_4_0_1_150_labels.txt"
test_file="../split_files/test_multi_label_thresh_0_4_0_1_150_labels.txt"

#load the lines in train,val and test 

train_lines=open(train_file).readlines()
val_lines=open(val_file).readlines()
test_lines=open(test_file).readlines()

#remove newline from entries 
train_lines=[x.strip() for x in train_lines]
val_lines=[x.strip() for x in val_lines]
test_lines=[x.strip() for x in test_lines]

#train, val, test ids overlap
train_ids=[t.split(" ")[0].split("/")[0] for t in train_lines]
val_ids=[t.split(" ")[0].split("/")[0] for t in val_lines]
test_ids=[t.split(" ")[0].split("/")[0] for t in test_lines]

#intersection between ids 
train_val_ids=list(set(train_ids).intersection(set(val_ids)))
train_test_ids=list(set(train_ids).intersection(set(test_ids)))
val_test_ids=list(set(val_ids).intersection(set(test_ids)))

#print some samples
# print("train_val_ids",train_ids[:10])
# print("train_test_ids",train_ids[:10])
# print("val_test_ids",val_ids[:10])

print("train_val_ids",len(train_val_ids))
print("train_test_ids",len(train_test_ids))
print("val_test_ids",len(val_test_ids))


#read the eval file 
#/data/digbose92/ambience_detection/codes/mica-MovieCLIP/split_files/test_shots_human_verified_multi_labels_v1_complete.txt
eval_file="../split_files/test_shots_human_verified_multi_labels_v1_complete.txt"
eval_lines=open(eval_file).readlines()
eval_lines=[x.strip() for x in eval_lines]
eval_vidsitu_ids=[t.split(" ")[0].split("/")[0] for t in eval_lines]

eval_id_sample=[]

for id in eval_vidsitu_ids:
    seg_location=id.index('seg')
    eval_id_sample.append(id[2:seg_location-1])
    #print(id[2:seg_location-1])
#print(eval_vidsitu_ids[0:10])

#intersection between train and eval ids 
train_eval_ids=list(set(train_ids).intersection(set(eval_id_sample)))
val_eval_ids=list(set(val_ids).intersection(set(eval_id_sample)))
test_eval_ids=list(set(test_ids).intersection(set(eval_id_sample)))


print("train_eval_ids",len(train_eval_ids))
print("val_eval_ids",len(val_eval_ids))
print("test_eval_ids",len(test_eval_ids))