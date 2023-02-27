#helper scripts to check if the shot segment files overlap in two directories

import os
from tqdm import tqdm

def generate_file_name(file_path):

    #print(file_path)
    scenes_index=file_path.index("Scenes")
    file_name=file_path[0:scenes_index-1]
    return(file_name)

def generate_cmd_clip_list(folder_list,base_folder):

    #print(folder_list)
    cmd_clip_list=[]
    for folder in tqdm(folder_list):
        folder_path=os.path.join(base_folder,folder)
        file_list=os.listdir(folder_path)
        file_list=[os.path.join(folder_path,s) for s in file_list]
        cmd_clip_list=cmd_clip_list+file_list
       
    return(cmd_clip_list)
    

shot_segment_file_v1="/data/digbose92/ambience_detection/codes/shot_segments/shot_segments_v1"
shot_segment_file_v2="/data/digbose92/ambience_detection/codes/shot_segments/shot_segments_v2"
folder="/data/ambience/Condensed_Movies/video_clips_shots_complete"
shot_subfolder=os.listdir(folder)
print(len(set(shot_subfolder)))

#scene file list v1 and v2
scene_file_list_v1=os.listdir(shot_segment_file_v1)
scene_file_list_v1=[s for s in scene_file_list_v1 if s.endswith(".csv")]
scene_file_list_v2=os.listdir(shot_segment_file_v2)

#total scene file list
total_scene_file_list=scene_file_list_v1+scene_file_list_v2
total_scene_file_list.remove("Nan_label_top_250.csv")
print(len(total_scene_file_list),len(set(total_scene_file_list)))

#number of total scene files
total_files=32484

print('Total number of scene files in v1+v2: ',len(scene_file_list_v1)+len(scene_file_list_v2))
print('Remaining files: ',total_files-len(scene_file_list_v1)-len(scene_file_list_v2))

#check intersection between two lists
intersection_list=list(set(scene_file_list_v1).intersection(scene_file_list_v2))
print(len(intersection_list)) #currently zero

#read the list of mkv files in the Condensed movies directory
CMD_clip_file="/data/ambience/Condensed_Movies/Condensed_Movies_downloaded_data/clip_list.txt"
with open(CMD_clip_file,'r') as f:
    CMD_clip_list=f.readlines()

CMD_clip_list_sample=[c.split("\n")[0].split("/")[-1] for c in CMD_clip_list]
#print(CMD_clip_list[0:5])

cnt_present_folder=0 #should be 28613
cnt_mkv_files=0 #should be 28613

subfold_present_list=[]
for scene_file in tqdm(total_scene_file_list):

    subfolder_name=generate_file_name(scene_file)
    subfold_present_list.append(subfolder_name)
    index_subfold=shot_subfolder.index(subfolder_name)
    cnt_present_folder+=1
    # except:
    #     print('Here')
    #     mkv_filename=subfolder_name+".mkv"
    #     print(mkv_filename)
    #     if mkv_filename in CMD_clip_list:
    #         cnt_mkv_files+=1
difference_folder=list(set(shot_subfolder)-set(subfold_present_list))
filename_incomplete_list=[]
for diff_fold in difference_folder:
    mkv_filename=diff_fold+".mkv"
    if mkv_filename in CMD_clip_list_sample:
        cnt_mkv_files+=1
        filename_incomplete_list.append(CMD_clip_list[CMD_clip_list_sample.index(mkv_filename)].split("\n")[0])
    # else:
    #     print(mkv_filename)

# print(cnt_present_folder)
print(cnt_mkv_files)
print(len(difference_folder)-cnt_mkv_files)
print(filename_incomplete_list)


with open('../data/shots_rerun_incomplete_list.txt','w') as f:
    for item in filename_incomplete_list:
        f.write("%s \n" % item)


# base_folder="/data/ambience/Condensed_Movies/Condensed_Movies_downloaded_data"
# folder_list=['2011','2012','2013','2014','2015','2016','2017','2018','2019']
# cmd_clip_list=generate_cmd_clip_list(folder_list,base_folder)#32333

#print(len(cmd_clip_list))