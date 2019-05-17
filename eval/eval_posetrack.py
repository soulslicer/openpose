import cv2
import numpy as np
import os
import json
import sys
import shutil
import subprocess
import sys
dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"

VALIDATION_PATH = sys.argv[1]
POSETRACK_RESULTS_PATH = dir_path + "posetrack_results/"
POSETRACK_PATH = dir_path + "posetrack/"
TRACKING = int(sys.argv[2])
MODEL = sys.argv[3]

#TRACKING = False

def load_json(path):
    if path.endswith(".json"):
        with open(path) as json_data:
            #print path
            d = json.load(json_data)
            json_data.close()
            return d
    return 0

def save_json(name, obj):
    with open(name, 'w') as outfile:
        json.dump(obj, outfile)

def get_anno_gt(frame_data):
    # Data
    img_name = frame_data["image"]["name"]
    imgnum = frame_data["imgnum"]
    is_labeled = frame_data["is_labeled"]
    #img = cv2.imread(POSETRACK_PATH + img_name)

    if is_labeled:
        annorects = frame_data["annorect"]
        if type(annorects) is dict:
            annorects = []
            annorects.append(frame_data["annorect"])

        # Anno object
        anno = dict()
        anno["cid"] = imgnum
        anno["image"] = [{"name":img_name}]
        anno["annorect"] = []

        # Hack it
        hacked_annorects = []
        for annorect in annorects:
            new_annorect = annorect
            if "annopoints" not in annorect:
                continue
            # Issues
            if len(annorect["annopoints"]) == 0:
                continue
            annopoints = annorect["annopoints"]["point"]
            if type(annopoints) is dict:
                annopoints_temp = annopoints
                annopoints = []
                annopoints.append(annopoints_temp)
            new_annorect["annopoints"]["point"] = annopoints

            for point in new_annorect["annopoints"]["point"]:
                point["id"] = [point["id"]]
                point["x"] = [point["x"]]
                point["y"] = [point["y"]]
                point["score"] = [1.0]
            new_annorect["track_id"] = [new_annorect["track_id"]]
            new_annorect["annopoints"] = [new_annorect["annopoints"]]
            try:
                new_annorect["score"] = [new_annorect["score"]]
            except:
                new_annorect["score"] = 1

            new_annorect["x1"] = [new_annorect["x1"]]
            new_annorect["y1"] = [new_annorect["y1"]]
            new_annorect["x2"] = [new_annorect["x2"]]
            new_annorect["y2"] = [new_annorect["y2"]]

            hacked_annorects.append(new_annorect)

        anno["annorect"] = hacked_annorects
        return anno

    return None

def op_to_pt(op_keypoints):

    POSETRACK_MAPPING = dict()
    POSETRACK_MAPPING["RANKLE"] = 0
    POSETRACK_MAPPING["RKNEE"] = 1
    POSETRACK_MAPPING["RHIP"] = 2
    POSETRACK_MAPPING["LHIP"] = 3
    POSETRACK_MAPPING["LKNEE"] = 4
    POSETRACK_MAPPING["LANKLE"] = 5
    POSETRACK_MAPPING["RWRIST"] = 6
    POSETRACK_MAPPING["RELBOW"] = 7
    POSETRACK_MAPPING["RSHOULDER"] = 8
    POSETRACK_MAPPING["LSHOULDER"] = 9
    POSETRACK_MAPPING["LELBOW"] = 10
    POSETRACK_MAPPING["LWRIST"] = 11
    POSETRACK_MAPPING["NECK"] = 12
    POSETRACK_MAPPING["NOSE"] = 13
    POSETRACK_MAPPING["TOP"] = 14

    if MODEL == "BODY_25B":
        BODY25B_MAPPING = dict()
        BODY25B_MAPPING["NOSE"] = 0
        BODY25B_MAPPING["LEYE"] = 1
        BODY25B_MAPPING["REYE"] = 2
        BODY25B_MAPPING["LEAR"] = 3
        BODY25B_MAPPING["REAR"] = 4
        BODY25B_MAPPING["LSHOULDER"] = 5
        BODY25B_MAPPING["RSHOULDER"] = 6
        BODY25B_MAPPING["LELBOW"] = 7
        BODY25B_MAPPING["RELBOW"] = 8
        BODY25B_MAPPING["LWRIST"] = 9
        BODY25B_MAPPING["RWRIST"] = 10
        BODY25B_MAPPING["LHIP"] = 11
        BODY25B_MAPPING["RHIP"] = 12
        BODY25B_MAPPING["LKNEE"] = 13
        BODY25B_MAPPING["RKNEE"] = 14
        BODY25B_MAPPING["LANKLE"] = 15
        BODY25B_MAPPING["RANKLE"] = 16
        BODY25B_MAPPING["UPPERNECK"] = 17
        BODY25B_MAPPING["HEADTOP"] = 18
        BODY25B_MAPPING["LBIGTOE"] = 19
        BODY25B_MAPPING["LSMALLTOE"] = 20
        BODY25B_MAPPING["LHEEL"] = 21
        BODY25B_MAPPING["RBIGTOE"] = 22
        BODY25B_MAPPING["RSMALLTOE"] = 23
        BODY25B_MAPPING["RHEEL"] = 24

        BODY25B_TO_POSETRACK = dict()
        BODY25B_TO_POSETRACK[BODY25B_MAPPING["RANKLE"]] = POSETRACK_MAPPING["RANKLE"]
        BODY25B_TO_POSETRACK[BODY25B_MAPPING["RKNEE"]] = POSETRACK_MAPPING["RKNEE"]
        BODY25B_TO_POSETRACK[BODY25B_MAPPING["RHIP"]] = POSETRACK_MAPPING["RHIP"]
        BODY25B_TO_POSETRACK[BODY25B_MAPPING["LHIP"]] = POSETRACK_MAPPING["LHIP"]
        BODY25B_TO_POSETRACK[BODY25B_MAPPING["LKNEE"]] = POSETRACK_MAPPING["LKNEE"]
        BODY25B_TO_POSETRACK[BODY25B_MAPPING["LANKLE"]] = POSETRACK_MAPPING["LANKLE"]
        BODY25B_TO_POSETRACK[BODY25B_MAPPING["RWRIST"]] = POSETRACK_MAPPING["RWRIST"]
        BODY25B_TO_POSETRACK[BODY25B_MAPPING["LELBOW"]] = POSETRACK_MAPPING["LELBOW"]
        BODY25B_TO_POSETRACK[BODY25B_MAPPING["RELBOW"]] = POSETRACK_MAPPING["RELBOW"]
        BODY25B_TO_POSETRACK[BODY25B_MAPPING["RSHOULDER"]] = POSETRACK_MAPPING["RSHOULDER"]
        BODY25B_TO_POSETRACK[BODY25B_MAPPING["LSHOULDER"]] = POSETRACK_MAPPING["LSHOULDER"]
        BODY25B_TO_POSETRACK[BODY25B_MAPPING["LWRIST"]] = POSETRACK_MAPPING["LWRIST"]
        BODY25B_TO_POSETRACK[BODY25B_MAPPING["UPPERNECK"]] = POSETRACK_MAPPING["NECK"]
        BODY25B_TO_POSETRACK[BODY25B_MAPPING["NOSE"]] = POSETRACK_MAPPING["NOSE"]
        BODY25B_TO_POSETRACK[BODY25B_MAPPING["HEADTOP"]] = POSETRACK_MAPPING["TOP"]

        pt_keypoints = np.zeros((15,3))
        for i in range(0, op_keypoints.shape[0]):
            if i in BODY25B_TO_POSETRACK.keys():
                pt_keypoints[BODY25B_TO_POSETRACK[i],:] = op_keypoints[i,:]

        return pt_keypoints

    elif MODEL == "BODY_21A":

        COCO21_MAPPING = dict()
        COCO21_MAPPING["NOSE"] = 0
        COCO21_MAPPING["NECK"] = 1
        COCO21_MAPPING["RSHOULDER"] = 2
        COCO21_MAPPING["RELBOW"] = 3
        COCO21_MAPPING["RWRIST"] = 4
        COCO21_MAPPING["LSHOULDER"] = 5
        COCO21_MAPPING["LELBOW"] = 6
        COCO21_MAPPING["LWRIST"] = 7
        COCO21_MAPPING["LOWERABS"] = 8
        COCO21_MAPPING["RHIP"] = 9
        COCO21_MAPPING["RKNEE"] = 10
        COCO21_MAPPING["RANKLE"] = 11
        COCO21_MAPPING["LHIP"] = 12
        COCO21_MAPPING["LKNEE"] = 13
        COCO21_MAPPING["LANKLE"] = 14
        COCO21_MAPPING["REYE"] = 15
        COCO21_MAPPING["LEYE"] = 16
        COCO21_MAPPING["REAR"] = 17
        COCO21_MAPPING["LEAR"] = 18
        COCO21_MAPPING["REALNECK"] = 19
        COCO21_MAPPING["TOP"] = 20

        COCO21_TO_POSETRACK = dict()
        COCO21_TO_POSETRACK[COCO21_MAPPING["RANKLE"]] = POSETRACK_MAPPING["RANKLE"]
        COCO21_TO_POSETRACK[COCO21_MAPPING["RKNEE"]] = POSETRACK_MAPPING["RKNEE"]
        COCO21_TO_POSETRACK[COCO21_MAPPING["RHIP"]] = POSETRACK_MAPPING["RHIP"]
        COCO21_TO_POSETRACK[COCO21_MAPPING["LHIP"]] = POSETRACK_MAPPING["LHIP"]
        COCO21_TO_POSETRACK[COCO21_MAPPING["LKNEE"]] = POSETRACK_MAPPING["LKNEE"]
        COCO21_TO_POSETRACK[COCO21_MAPPING["LANKLE"]] = POSETRACK_MAPPING["LANKLE"]
        COCO21_TO_POSETRACK[COCO21_MAPPING["RWRIST"]] = POSETRACK_MAPPING["RWRIST"]
        COCO21_TO_POSETRACK[COCO21_MAPPING["RELBOW"]] = POSETRACK_MAPPING["RELBOW"]
        COCO21_TO_POSETRACK[COCO21_MAPPING["RSHOULDER"]] = POSETRACK_MAPPING["RSHOULDER"]
        COCO21_TO_POSETRACK[COCO21_MAPPING["LSHOULDER"]] = POSETRACK_MAPPING["LSHOULDER"]
        COCO21_TO_POSETRACK[COCO21_MAPPING["LELBOW"]] = POSETRACK_MAPPING["LELBOW"]
        COCO21_TO_POSETRACK[COCO21_MAPPING["LWRIST"]] = POSETRACK_MAPPING["LWRIST"]
        COCO21_TO_POSETRACK[COCO21_MAPPING["REALNECK"]] = POSETRACK_MAPPING["NECK"]
        COCO21_TO_POSETRACK[COCO21_MAPPING["NOSE"]] = POSETRACK_MAPPING["NOSE"]
        COCO21_TO_POSETRACK[COCO21_MAPPING["TOP"]] = POSETRACK_MAPPING["TOP"]

        pt_keypoints = np.zeros((15,3))
        for i in range(0, op_keypoints.shape[0]):
            if i in COCO21_TO_POSETRACK.keys():
                pt_keypoints[COCO21_TO_POSETRACK[i],:] = op_keypoints[i,:]

        return pt_keypoints


def get_anno_op(frame_data, frame_op_data):
    # Data
    img_name = frame_data["image"]["name"]
    imgnum = frame_data["imgnum"]
    is_labeled = frame_data["is_labeled"]
    #img = cv2.imread(POSETRACK_PATH + img_name)

    if is_labeled:
        annorects = frame_data["annorect"]
        if type(annorects) is dict:
            annorects = []
            annorects.append(frame_data["annorect"])

        # Anno object
        anno = dict()
        anno["cid"] = imgnum
        anno["image"] = [{"name":img_name}]
        anno["annorect"] = []

        # Convert OP to PT
        pt_detections = []
        pt_tids = []
        for person_op in frame_op_data["people"]:
            op_keypoints = np.array(person_op["pose_keypoints_2d"])
            op_keypoints = op_keypoints.reshape((op_keypoints.shape[0]/3, 3))
            pt_keypoints = op_to_pt(op_keypoints)
            pt_detections.append(pt_keypoints)
            if TRACKING: pt_tids.append(person_op["person_id"][0])
        if not TRACKING: pt_tids = np.arange(len(pt_detections))

        new_annorects = []
        for pt_detect, pt_tid in zip(pt_detections, pt_tids):
            # if pt_tid in to_remove_tids:
            #     continue

            new_annorect = dict()
            new_annorect["annopoints"] = dict()
            new_annorect["annopoints"]["point"] = []

            # PT Rect Compute
            pt_points = []
            for j in range(pt_detect.shape[0]):
                if pt_detect[j,2] < 0.05: continue
                point = dict()
                point["id"] = [j]
                point["x"] = [pt_detect[j,0]]
                point["y"] = [pt_detect[j,1]]
                point["score"] = [pt_detect[j,2]]
                new_annorect["annopoints"]["point"].append(point)

            new_annorect["track_id"] = [pt_tid]
            new_annorect["annopoints"] = [new_annorect["annopoints"]]
            #new_annorect["annopoints"] = [hacked_annorects[i]["annopoints"]]
            new_annorect["score"] = [1.0]
            new_annorects.append(new_annorect)

        anno["annorect"] = new_annorects
        return anno

    return None


def create_gt():
    # Recreate Folder
    if os.path.exists(POSETRACK_RESULTS_PATH + "eval_gt"):
        shutil.rmtree(POSETRACK_RESULTS_PATH + "eval_gt")
    os.makedirs(POSETRACK_RESULTS_PATH + "eval_gt")

    # Load Jsons
    val_jsons = []
    for filename in sorted(os.listdir(VALIDATION_PATH)):
        val_jsons.append([load_json(VALIDATION_PATH + filename), filename])

    # Iterate Each Video
    for video_data, filename in val_jsons:
        output = dict()
        output["annolist"] = []

        #if filename != "016236_mpii_test.json": continue

        # Iterate each frame
        for i in range(0, len(video_data)):
            frame_data = video_data[i]

            # Ground Truth Frame
            anno_gt = get_anno_gt(frame_data)
            if anno_gt is not None:
                output["annolist"].append(anno_gt)

        # Write to Disk
        save_json(POSETRACK_RESULTS_PATH + "eval_gt/" + filename, output)

def create_op():
    # Recreate Folder
    if os.path.exists(POSETRACK_RESULTS_PATH + "eval_op"):
        shutil.rmtree(POSETRACK_RESULTS_PATH + "eval_op")
    os.makedirs(POSETRACK_RESULTS_PATH + "eval_op")

    # Load Jsons
    val_jsons = []
    for filename in sorted(os.listdir(VALIDATION_PATH)):
        val_jsons.append([load_json(VALIDATION_PATH + filename), filename])

    # Iterate Each Video
    for video_data, filename in val_jsons:
        output = dict()
        output["annolist"] = []

        #if filename != "016236_mpii_test.json": continue

        # Iterate each frame
        for i in range(0, len(video_data)):
            frame_data = video_data[i]

            # Grab OP Frame
            frame_op_data = load_json(POSETRACK_RESULTS_PATH + "op_output/" + filename.split(".")[0] + "/" + "%06d" % (i) + "_keypoints.json")
            anno_op = get_anno_op(frame_data, frame_op_data)
            if anno_op is not None:
                output["annolist"].append(anno_op)

        # Write to Disk
        save_json(POSETRACK_RESULTS_PATH + "eval_op/" + filename, output)

def evaluate():
    cmd = "cd "+dir_path+"/posetrack/posetrack_valscripts/py; python evaluate.py -g "+POSETRACK_RESULTS_PATH + "eval_gt/"+" -p "+POSETRACK_RESULTS_PATH + "eval_op/"+" -e"
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).stdout.read()
    print(output)
    if TRACKING:
        cmd = "cd "+dir_path+"/posetrack/posetrack_valscripts/py; python evaluate.py -g "+POSETRACK_RESULTS_PATH + "eval_gt/"+" -p "+POSETRACK_RESULTS_PATH + "eval_op/"+" -t"
        output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).stdout.read()
        print(output)        

create_gt()
create_op()
evaluate()
