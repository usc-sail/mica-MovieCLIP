import os 
import json 
import pandas as pd 
import numpy as np 
import argparse
from tqdm import tqdm
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('--source_file', type=str, required=True)
parser.add_argument('--destination_folder', type=str, required=True)


args = parser.parse_args()
source_file = args.source_file
dest_folder = args.destination_folder

#read the json file
with open(source_file,'r') as f:
    data=json.load(f)

video_keys=list(data.keys())
total_labels_list=[]

for video_key in tqdm(list(video_keys)):

    video_data=data[video_key]

    for shot_name in list(video_data.keys()):
        shot_data=video_data[shot_name]
        labels=shot_data['labels']
        shot_labels={l for l in list(labels.keys()) if labels[l]>=0.4}
        total_labels_list=total_labels_list+list(shot_labels)

        #print(shot_data['labels'])
total_labels_counter=Counter(total_labels_list)
total_labels_dict=dict(total_labels_counter)
print(len(total_labels_dict))

#save the total labels dict
with open(os.path.join(dest_folder,'clean_labels_movieCLIP_distribution.json'),'w') as f:
    json.dump(total_labels_dict,f,indent=4)



        
