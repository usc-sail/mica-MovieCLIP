import os 
import pandas as pd 
import json 
import argparse
from tqdm import tqdm  
import pickle
from collections import Counter
#json file with the keys as video id and sub keys as the Scene Number 
#for each scene number there is a sub-dictionary with labels and scores, Start time and end time 

# total tags list: 32485 clips and associated shots
# current available tags list: 32361 clips and associated shots
columns=['Scene Number', 'Start Frame', 'Start Timecode', 'Start Time (seconds)',
           'End Frame', 'End Timecode', 'End Time (seconds)', 'Length (frames)',
           'Length (timecode)', 'Length (seconds)']

def read_shot_segment_csv_file(csv_file_path):
    df=pd.read_csv(csv_file_path,skiprows=1)
    return df

def generate_video_key_wise_dictionary(file_key,csv_data,tag_data):
    #print(csv_data['Scene Number'])
    #print(tag_data.keys())
    #format for key in tag_data is <file_key>-Scene-<scene_number>.mp4
    #print(len(tag_data),len(csv_data['Scene Number']))
    #assert(len(tag_data)==len(csv_data['Scene Number']))
    shot_level_dict={}
    num_clean_samples_video=0
    total_set_labels=[]


    for i in range(len(csv_data['Scene Number'])):
        
        #scene number + start time + end time
        scene_number=csv_data['Scene Number'].iloc[i]
        start_time=csv_data['Start Time (seconds)'].iloc[i]
        end_time=csv_data['End Time (seconds)'].iloc[i]
        start_frame=csv_data['Start Frame'].iloc[i]
        end_frame=csv_data['End Frame'].iloc[i]


        format_number_string="{:03d}".format(scene_number)
        key_name=file_key+"-Scene-"+str(format_number_string)+".mp4"

        if(key_name in tag_data):
            scene_data=tag_data[key_name]

            #labels + scores 
            labels_list=scene_data['Labels']
            scores_list=scene_data['Values']
            if(len(scores_list)>0):
                if(scores_list[0]>=0.4):
                    num_clean_samples_video+=1
                    total_set_labels=total_set_labels+labels_list

            #create a dict with the labels_list and scores_list
            label_dict={ labels_list[i]:scores_list[i] for i in range(len(labels_list))}

            #dict for the key 
            temp_dict={'start_frame':float(start_frame),
                       'end_frame':float(end_frame),
                       'start_time':start_time,
                       'end_time':end_time,
                        'labels': label_dict
                    }
            
            shot_level_dict[key_name]=temp_dict

    return(shot_level_dict,num_clean_samples_video,total_set_labels)

parser=argparse.ArgumentParser()
parser.add_argument('--destination_folder', type=str, required=True)
parser.add_argument('--shot_tags_folder',type=str, required=True)
parser.add_argument('--video_shots_folder',type=str, required=True)

args=parser.parse_args()
dest_folder=args.destination_folder
shot_tags_folder=args.shot_tags_folder
video_shots_folder=args.video_shots_folder

#obtain the path of segments file 
shot_segments_v1=os.path.join(video_shots_folder,'shot_segments_v1')
shot_segments_v2=os.path.join(video_shots_folder,'shot_segments_v2')
shot_segments_v3=os.path.join(video_shots_folder,'shot_segments_v3')

shot_segments_v1_files=[os.path.join(shot_segments_v1,f) for f in os.listdir(shot_segments_v1) if f.endswith('.csv')]
shot_segments_v2_files=[os.path.join(shot_segments_v2,f) for f in os.listdir(shot_segments_v2) if f.endswith('.csv')]
shot_segments_v3_files=[os.path.join(shot_segments_v3,f) for f in os.listdir(shot_segments_v3) if f.endswith('.csv')]

#total list of segment files
total_segment_files=shot_segments_v1_files+shot_segments_v2_files+shot_segments_v3_files
total_segment_files=list(set(total_segment_files))
# print(len(total_segment_files),len(set(total_segment_files)))
# print(total_segment_files[0:2])
total_tagged_files_present=0
empty_tagged_files=0
total_clean_samples_video=0

#list of tagged files
tagged_file_list=os.listdir(shot_tags_folder)

not_present_list=[]
empty_tagged_file_list=[]
num_not_equal_files=0
equal_files=0
not_equal_list=[]
total_tagged_samples=0
movieCLIP_dict={}
num_samples=0
total_labels=[]

for file in tqdm(total_segment_files):
    
    #read the csv file and the different shot segments 
    shot_csv_data=read_shot_segment_csv_file(file)
    
    #file key here
    file_key=file.split("/")[-1]
    
    #tagged file key and pkl file name
    tag_file_name=file_key.split("-Scenes")[0]
    pkl_file_name=tag_file_name+".pkl"

    if(pkl_file_name in tagged_file_list):
        total_tagged_files_present+=1
        pkl_file=os.path.join(shot_tags_folder,pkl_file_name)
        #print(pkl_file,file,tag_file_name)

        with open(pkl_file,'rb') as f:
            clip_tag_data=pickle.load(f)

        if(len(clip_tag_data)==0):
            empty_tagged_files+=1
            empty_tagged_file_list.append(pkl_file_name)

        else:
           
            #generate_video_key_wise_dictionary(tag_file_name,shot_csv_data,clip_tag_data)
            shot_level_dict,total_clean_samples_per_video,total_clean_labels_list=generate_video_key_wise_dictionary(tag_file_name,shot_csv_data,clip_tag_data)
            total_clean_samples_video+=total_clean_samples_per_video
            total_labels=total_labels+total_clean_labels_list #total clean labels list
            
            movieCLIP_dict[tag_file_name]=shot_level_dict
            
            total_tagged_samples+=len(shot_level_dict)
            equal_files+=1
            num_samples+=1
    else:
        not_present_list.append(pkl_file_name)

    # if(num_samples==2):
    #     break

print("Total tagged files present: ",total_tagged_files_present) #32357
print("Total empty tagged files: ",empty_tagged_files) #0
print("Total files not equal in length:",num_not_equal_files) #128
print("Total files equal: ",equal_files) #0
print("Total tagged samples: ",total_tagged_samples) #0
print("Total clean samples: ",total_clean_samples_video) #0

# #save the empty tagged files list
# with open(os.path.join(dest_folder,'empty_tagged_file_list.pkl'),'wb') as f:
#     pickle.dump(empty_tagged_file_list,f)

# #save the not present list
# with open(os.path.join(dest_folder,'not_equal_list.pkl'),'wb') as f:
#     pickle.dump(not_equal_list,f)

# print the distribution of the labels
total_clean_label_counter=Counter(total_labels)
total_clean_labels_dict=dict(total_clean_label_counter)

#save the dictionary as a json
with open(os.path.join(dest_folder,'movieCLIP_dataset_class_clean_distribution.json'),'w') as f:
    json.dump(total_clean_labels_dict,f,indent=4)

#print(total_clean_label_counter)

#plot the distribution of the labels using a




#save the dictionary as a json
#print(movieCLIP_dict)
# with open(os.path.join(dest_folder,'movieCLIP_dataset.json'),'w') as f:
#     json.dump(movieCLIP_dict,f,indent=4)
#Total tagged samples:  1116190
# print(not_present_list)
#Total clean samples:  118771 (threshold >=0.4)
#extra files like ['csv_files.pkl', 'extract_scenes_condensed_movies_clips.py.pkl', 'Nan_label_top_250.csv.pkl', 'extract_shots_condensed_movies.py.pkl']




