import os 
import pandas as pd 
import numpy as np 
from tqdm import tqdm 
import pickle
import multiprocessing as mp
import argparse 

destination_scenes_folder='/data/ambience/Condensed_Movies/video_clips_shots_complete'
csv_scenes_folder="/data/ambience/Condensed_Movies/video_clips_shots_stats_complete"
file_list_pickle_file="/data/digbose92/ambience_detection/pkl-files/Condensed_Movies_updated_list_large_set.pkl"

with open(file_list_pickle_file,"rb") as f:
    file_list=pickle.load(f)

# def extract_scene_clips(idx):
#     vid_file=file_list[idx]
#     file_key=vid_file.split("/")[-1][:-4]
#     subfolder=os.path.join(destination_scenes_folder,file_key)
#     csv_scenes_file=os.path.join(csv_scenes_folder,file_key+".csv")

#     if(os.path.exists(csv_scenes_file) is False):
#         os.mkdir(subfolder)
#         scene_detect_command="scenedetect --input "+vid_file+ " -s "+csv_scenes_file+" detect-content list-scenes split-video -o "+subfolder
#         os.system(scene_detect_command)
def extract_scene_clips(idx):
    vid_file=file_list[idx]
    file_key=vid_file.split("/")[-1][:-4]
    subfolder=os.path.join(destination_scenes_folder,file_key)
    csv_scenes_file=os.path.join(csv_scenes_folder,file_key+".csv")

    if(os.path.exists(subfolder) is False):
        os.mkdir(subfolder)
        scene_detect_command="scenedetect --input "+vid_file+ " -s "+csv_scenes_file+" detect-content list-scenes split-video -o "+subfolder
        os.system(scene_detect_command)

#condensed_movies_folder='/data/ambience/Condensed_Movies/video_clips_downsampled'


# #print(len(condensed_movies_folder))
# for vid_file in tqdm(file_list):
#     file_key=vid_file.split("/")[-1][:-4]
#     subfolder=os.path.join(destination_scenes_folder,file_key)
#     csv_scenes_file=os.path.join(csv_scenes_folder,file_key+".csv")

#     if(os.path.exists(csv_scenes_file) is False):
#         os.mkdir(subfolder)
#         scene_detect_command="scenedetect --input "+vid_file+ " -s "+csv_scenes_file+" detect-content list-scenes split-video -o "+subfolder
#         os.system(scene_detect_command)
        #print(scene_detect_command)
    #scene_det
    #print(subfolder)
    #print(file_key)
def main(args):
#    data = [x.rstrip().split(',') for x in open('/data/movies/movie_sounds_50_mturk_test.csv').readlines()[1:]]
#   data = [x.rstrip().split(',') for x in open('/data/rajatheb/sound_events/isound_event_labels.csv').readlines()[1:]]
    pool = mp.Pool(args.nj)
    pool.map(extract_scene_clips, list(range(len(file_list))))
    pool.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--nj', default=16, type=int, help='number of parallel processes')
    args = parser.parse_args()
    main(args)
