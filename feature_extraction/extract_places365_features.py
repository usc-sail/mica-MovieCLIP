#from dataset_HVU_feature import *
import torchvision.models as models
import torch
import cv2
from torchvision import transforms as trn
from torch.nn import functional as F
import time 
import numpy as np 
from tqdm import tqdm
import os 
import pickle
#from pims import PyAVVideoReader
from PIL import Image
import math
import pandas as pd 
import time 
import torch.nn as nn

activation = {}
def getActivation(name):
  # the hook signature
  def hook(model, input, output):
    activation[name] = output.detach()
  return hook

def generate_video_prediction(model,device,video_file,trn,batch_size=4):

    vcap=cv2.VideoCapture(video_file)
    feature_list=np.zeros((0,512))
    frame_id=1
    length = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_list=[]
    block_count=0
    #print(video_file)
    while True:
        ret, frame = vcap.read()
        if(ret==True):
            
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #convert BGR to RGB
            frame=trn(Image.fromarray(frame)) #convert BGR image to PIL Image
            frame =frame.unsqueeze(0).to(device) #preprocess the frame


            #concatenate till batch size is divisible by 32 and then do a forward pass 
            if((frame_id%batch_size==0) or (frame_id==length)):
                frame_list.append(frame)
                frame_comb=torch.cat(frame_list,dim=0)
                block_count=block_count+1
                #extract the image features
                #print(frame_comb.size())
                with torch.no_grad():
                    image_features = model.forward(frame_comb)

                frame_list=[]
                avgpool_feat=activation['avgpool'].squeeze(2).detach().cpu()
                avgpool_feat=avgpool_feat.squeeze(2).numpy()
                #image_features=image_features.cpu().numpy()
                feature_list=np.vstack([feature_list,avgpool_feat])
                #print(frame_id,feature_list.shape)
                #add this chunk to an empty array 
            else:
                frame_list.append(frame)
                
            #print(frame_id,len(frame_list))
            #image_features=image_features.cpu().numpy().squeeze(0)
            #feature_list.append(image_features)
            
            frame_id=frame_id+1
        else:
            #print(frame_id+1)
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #print(block_count,feature_list.shape,length,frame_id)
    #image_features=np.array(feature_list)
    return(feature_list)







def generate_video_tensor_cv2(video_filename,trn,desired_frameRate=4):
    #print(os.path.exists(video_filename))
    cap=cv2.VideoCapture(video_filename)
    frameRate = cap.get(5)
    intfactor=math.ceil(frameRate/desired_frameRate)
    #print(intfactor)
    # desired_frameRate=
    #print(frameRate)
    tensor_list=[]
    frame_id=0
    while(cap.isOpened()):
        #frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if(ret):
            if (frame_id % intfactor == 0):
                frame=trn(Image.fromarray(frame))
                tensor_list.append(frame)
            frame_id=frame_id+1
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tensor_list=torch.stack(tensor_list)
    #print(frameRate,frameId,tensor_list.size()[0])
    return(tensor_list,frame_id)


def generate_predictions_frame_wise(model,vid_tensor):
    stack_vid_predictions=[]
    for i in np.arange(vid_tensor.shape[0]):
        vid_tensor_c=vid_tensor[i,:].unsqueeze(0)
        with torch.no_grad():
            image_features = model.forward(vid_tensor_c)
            avgpool_feat=activation['avgpool'].squeeze(2)
            avgpool_feat=avgpool_feat.squeeze(2)
            stack_vid_predictions.append(avgpool_feat)

    stack_vid_predictions=torch.stack(stack_vid_predictions)
    stack_vid_predictions=stack_vid_predictions.squeeze(1).cpu().numpy()
    return(stack_vid_predictions)



# pkl_file="../../hvu_video_dataset_labels/hvu_scene_extra_keys_feat_rem.pkl"
# feature_folder="/data/HVU/features/places365_features/train_features/places_365_resnet18_1_fps_train"
# folder="/data/HVU/mp4/train"

feature_folder="/data/ambience/VidSitu/features/places_365_4_fps"
base_folder="/data/ambience/VidSitu/shots"
#"/data/ambience/Condensed_Movies/features/places365_4_fps"
#csv_file="/data/digbose92/ambience_detection/csv_condensed_movies_shots_files/top_1_CLIP_labels_threshold_top_150_0.2.csv"
txt_file="/data/digbose92/ambience_detection/vidsitu_txt_files/test_shots_human_verified_multi_labels_v1_complete.txt"
with open(txt_file,"r") as f:
    lines=f.readlines()

file_names=[f.split("\n")[0] for f in lines]
file_name_set=[f.split(" ")[0] for f in lines]
file_name_set=[os.path.join(base_folder,f) for f in file_name_set]
#"/home/dbose_usc_edu/data/Condensed_Movies/clip_annotation_csv_files/top_1_CLIP_labels_threshold_top_150_gcp_0.2.csv"
#csv_data=pd.read_csv(csv_file)

centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

#hvu_video_dataset=HVU_Video_dataset(pkl_file=pkl_file,folder=folder,transform=centre_crop)
# th architecture to use
arch = 'resnet18'
#arch='densenet161'
# load the pre-trained weights
model_file = '%s_places365.pth.tar' % arch
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)
model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
state_dict = {str.replace(k,'norm.','norm'): v for k,v in state_dict.items()}
state_dict = {str.replace(k,'conv.','conv'): v for k,v in state_dict.items()}
state_dict = {str.replace(k,'normweight','norm.weight'): v for k,v in state_dict.items()}
state_dict = {str.replace(k,'normrunning','norm.running'): v for k,v in state_dict.items()}
state_dict = {str.replace(k,'normbias','norm.bias'): v for k,v in state_dict.items()}
state_dict = {str.replace(k,'convweight','conv.weight'): v for k,v in state_dict.items()}
# state_dict = {str.replace(k,'norm.','norm'): v for k,v in state_dict.items()}
# state_dict = {str.replace(k,'conv.','conv'): v for k,v in state_dict.items()}
# state_dict = {str.replace(k,'normweight','norm.weight'): v for k,v in state_dict.items()}
# state_dict = {str.replace(k,'normrunning','norm.running'): v for k,v in state_dict.items()}
# state_dict = {str.replace(k,'normbias','norm.bias'): v for k,v in state_dict.items()}
# state_dict = {str.replace(k,'convweight','conv.weight'): v for k,v in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()
#model=nn.DataParallel(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=model.to(device)
#print(model)
for name, layer in model.named_modules():
    print(name, layer)
#print(model.denselayer36[0].conv2)
#h1 = model.features.norm5.register_forward_hook(getActivation('norm5'))
h1 = model.avgpool.register_forward_hook(getActivation('avgpool'))
#for densenet161 this should be norm5
avgpool_list=[]
fail_list=[]

# with open(pkl_file,"rb") as f:
#     data_c=pickle.load(f)

# key_list=list(data_c.keys())

#load the failure file if it exists 
# filename_total_list=list(csv_data['Filename'])
# file_name_list=[f.split('/')[-1] for f in filename_total_list]
# filename_run=[]
# if os.path.exists('/data/digbose92/ambience_detection/pkl-files/failure_list.pkl') is True:
#     with open('/data/digbose92/ambience_detection/pkl-files/failure_list.pkl','rb') as f:
#         fail_list=pickle.load(f)
#     for f_t in fail_list:
#         if f_t in file_name_list:
#             index=file_name_list.index(f_t)
#             filename_run.append(filename_total_list[index])
# else:
#     filename_run=filename_total_list
    

#print(filename_run)

# fail_list=[]
# less_than_1_sec_list=[]
filename_run=file_name_set
for file in tqdm(filename_run):
    #data_curr=data_c[key]   
    #key_set=list(data_curr.keys())
    #if('frame_indices' in key_set):
    #try:
        #print(key)
        file_key=file.split("/")[-1][:-4]
        #print(file_key)
        numpy_filename=os.path.join(feature_folder,file_key+".npy")
        #if(os.path.exists(numpy_filename) is True):
            #print('Filename %s exists' %(numpy_filename))
        #else:
        if(os.path.exists(numpy_filename) is False):
            #print('Processing file:%s' %(numpy_filename))
            
            #vid_tensor,frame_cnt=generate_video_prediction(model,device,file,centre_crop,batch_size=4)
            vid_tensor,frame_cnt=generate_video_tensor_cv2(file,centre_crop,desired_frameRate=4)
            #generate_video_tensor_cv2(file,centre_crop)
            #print(vid_tensor.size(),frame_cnt)
            #vid_tensor=vid_tensor.squeeze(0)
            vid_tensor=vid_tensor.to(device)
            #print(vid_tensor.size())
            avgpool_feat=generate_predictions_frame_wise(model,vid_tensor)
            #print(vid_tensor.size())
            # start_time=time.time()
            # logit = model.forward(vid_tensor)
            # #print(logit.size())
            # #feature_list = generate_video_prediction(model,device,file,centre_crop,batch_size=4)
            # avgpool_feat=activation['avgpool'].squeeze(2).detach().cpu()
            # avgpool_feat=avgpool_feat.squeeze(2).numpy()

            #print(frame_cnt,avgpool_feat.shape)
            # end_time=time.time()
            # elapsed_time=end_time-start_time 
            #print(elapsed_time)
            #feat_set=activation['norm5'].detach().cpu()
            #print(feature_list.shape)
            
            #print(feat_set.size(),frame_rate)
            
            np.save(numpy_filename,avgpool_feat)
            del vid_tensor
            torch.cuda.empty_cache()
    # except:
    #     #print(vid_tensor.size())
    #     fail_list.append(file.split("/")[-1])
        

