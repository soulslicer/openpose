import cv2
import numpy as np
import os
import json
import sys
dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"
sys.path.insert(0, dir_path + "cocoapi-master/PythonAPI/")
from pycocotools import coco as cc
from pycocotools import cocoeval as ce
import glob
import os.path
import time

def test_single_scale():
    cocoGt = cc.COCO("annotations/person_keypoints_val2017.json")
    cocoDt = cocoGt.loadRes("testing_results/person_keypoints_test-val2017_OpenPose1scale_results.json");
    cocoEval = ce.COCOeval(cocoGt,cocoDt,'keypoints');
    cocoEval.params.imgIds = cocoGt.getImgIds();
    cocoEval.evaluate();
    cocoEval.accumulate();
    print("Single Scale")
    cocoEval.summarize();

test_single_scale()