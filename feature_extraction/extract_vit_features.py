import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch 
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import os 
import cv2 
import math
import argparse

activation = {}
def getActivation(name):
  # the hook signature
  def hook(model, input, output):
    activation[name] = output.detach()
  return hook

def run_frame_wise_feature_inference(model,transform,filename,device,dim=768,desired_frameRate=4):
  
    vcap=cv2.VideoCapture(filename)
    frameRate = vcap.get(5)
    intfactor=math.ceil(frameRate/desired_frameRate)
    feature_list=np.zeros((0,dim))
    frame_id=0
    length = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    tensor_list=[]
    while True:
        ret, frame = vcap.read()
        if(ret==True):
            if (frame_id % intfactor == 0):
                #print(frame_id)
                frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                frame=Image.fromarray(frame)
                tensor = transform(frame)
                tensor = tensor.to(device).unsqueeze(0) #convert each frame to tensor and pass to device 
                feat_tensor=model.forward_features(tensor) #pass tensor to the model and get the features
                feat_tensor=feat_tensor.cpu().detach().numpy() #convert the feature tensors to numpy array
                feature_list=np.vstack([feature_list,feat_tensor]) #add the features to the numpy array
                del tensor
                torch.cuda.empty_cache()
            frame_id=frame_id+1
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    return feature_list, frame_id


#declare vit models from timm specification
print('Loading model')
model = timm.create_model('vit_base_patch16_224', pretrained=True)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=model.to(device)
model.eval()

#declare the transforms
config = resolve_data_config({}, model=model)
transform = create_transform(**config)

print('Loaded model')
#load the model along with the logits
h1 = model.pre_logits.register_forward_hook(getActivation('pre_logits'))



#argparse arguments 


parser = argparse.ArgumentParser(description='Extract vit base features from a video file')
parser.add_argument('--feature_folder', type=str, help='path to the destination feature folder')
parser.add_argument('--base_video_folder', type=str, help='path to the base video folder')
parser.add_argument('--txt_file', type=str, help='path to the text file containing the video names')
#txt file will contain subfolder/filename
args = parser.parse_args()


#declaring the data 
feature_folder=args.feature_folder
base_video_folder=args.base_video_folder
txt_file=args.txt_file
cnt_processed=0
cnt_not_processed=0

with open(txt_file,"r") as f:
    lines=f.readlines()
file_names=[f.split("\n")[0] for f in lines]
file_name_set=[f.split(" ")[0] for f in lines]
file_name_set=[os.path.join(base_video_folder,f) for f in file_name_set]

for file in tqdm(file_name_set):
        file_list_c=file.split("/")
        file_key=file_list_c[-1]
        np_filename=file_key[:-4]+".npy"
        numpy_filename=os.path.join(feature_folder,np_filename)
        if(os.path.exists(numpy_filename)):
                continue
        else:
            cnt_not_processed+=1
            file=os.path.join(base_video_folder,file_list_c[-2],file_list_c[-1])
            frame_tensor,frame_id=run_frame_wise_feature_inference(model,transform,file,device)
            np.save(numpy_filename,frame_tensor)
            

print('Processed files:%d'%(cnt_processed))
print('Not processed files:%d'%(cnt_not_processed))