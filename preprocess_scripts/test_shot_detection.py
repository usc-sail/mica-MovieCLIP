import os 
import pandas as pd 
import numpy as np 
from tqdm import tqdm 
import pickle

destination_folder="/data/ambience/Condensed_Movies/test_shots_data"
video_file="/data/ambience/Condensed_Movies/Condensed_Movies_downloaded_data/2012/_8LrZ4NhPmk.mkv"
subfolder=os.path.join(destination_folder,os.path.splitext(video_file.split("/")[-1])[0])
csv_file="test.csv"
scene_detect_command="scenedetect --input "+video_file+ " -s "+csv_file+" detect-content list-scenes split-video -o "+subfolder

os.system(scene_detect_command)