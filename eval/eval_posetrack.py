import cv2
import numpy as np
import os
import json
import sys
dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"

def load_json(path):
    if path.endswith(".json"):
        with open(path) as json_data:
            #print path
            d = json.load(json_data)
            json_data.close()
            return d
    return 0

def get_anno(frame_data):
    

VALIDATION_PATH = dir_path + "posetrack/annotations/val_json/" 

# Load Jsons
val_jsons = []
for filename in sorted(os.listdir(VALIDATION_PATH)):
    val_jsons.append([load_json(VALIDATION_PATH + filename), filename])

# Iterate Each Video
for video_data, filename in val_jsons:

    # Iterate each frame
    for i in range(0, len(video_data)):
        frame_data = video_data[i]

        # Data
        img_name = frame_data["image"]["name"]
        imgnum = frame_data["imgnum"]
        is_labeled = frame_data["is_labeled"]
        img = cv2.imread(params["posetrack_data"] + img_name)

        if is_labeled:


    stop
    print filename

#load_json("")