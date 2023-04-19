import os
from re import I 
import pandas as pd 
import numpy as np 
from tqdm import tqdm
import clip 
import torch 
import cv2 
import pickle
from PIL import Image
import argparse
import json

def generate_shot_tags(shot_file_name,model,preprocess,text_features,device,label_list):

    #read individual frame and use CLIP individual encoder 
    vcap=cv2.VideoCapture(shot_file_name)
    similarity_list=[]
    while True:
        ret, frame = vcap.read()
        if(ret==True): # if it is a valid frame
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #convert BGR to RGB
            frame=Image.fromarray(frame) #convert BGR image to PIL Image
            frame = preprocess(frame).unsqueeze(0).to(device) #preprocess the frame
            with torch.no_grad():
                image_features = model.encode_image(frame)
    
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1) 
            similarity_list.append(similarity)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    similarity_score_tensor_frames=torch.cat(similarity_list,dim=0)

    similarity_score_tensor_scores=similarity_score_tensor_frames.mean(dim=0)
    
    values, indices = similarity_score_tensor_scores.topk(5)

    label_list_val=[label_list[id] for id in indices]
    val_list=[val.item() for val in values]
    
    return(label_list_val,val_list)


ap=argparse.ArgumentParser()
ap.add_argument('--label_file',required=True,help='path to the label file')
ap.add_argument('--source_folder',required=True,help='path to the source folder')
ap.add_argument('--output_folder',required=True,help='path to the output folder')

args_list=vars(ap.parse_args())

label_file=args_list['label_file']
source_folder=args_list['source_folder']

with open(label_file,'r') as f:
    label_list=f.readlines()

label_list=[label.strip().split("\n")[0] for label in label_list]

#initialize model and extract features for text labels through scene prompts
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}, a type of background location") for c in label_list]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
text_features /= text_features.norm(dim=-1, keepdim=True)

shot_subfolders=os.listdir(source_folder)

shot_dict={}

for shot_subfolder in tqdm(shot_subfolders):
    shot_subfolder_path=os.path.join(source_folder,shot_subfolder)
    shot_files=os.listdir(shot_subfolder_path)

    shot_temp_dict={}
    for shot_file in tqdm(shot_files):
        shot_file_path=os.path.join(shot_subfolder_path,shot_file)
        label_list_val,val_list=generate_shot_tags(shot_file_path,model,preprocess,text_features,device,label_list)
        
        #create a dictionary with the labels and the values
        shot_tags_dict={k:v for k,v in zip(label_list_val,val_list)}

        shot_temp_dict[shot_file]=shot_tags_dict

    shot_dict[shot_subfolder]=shot_temp_dict

output_file_name=os.path.join(args_list['output_folder'],'shot_tags.json')
with open(output_file_name,'w') as f:
    json.dump(shot_dict,f)