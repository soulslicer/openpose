import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
import sys
from shutil import copyfile
import time

HOME_MODEL = "/home/raaj/openpose_staf/models/pose/body_25b_video3/"
V100_MODEL = "/home/raaj/v100server2/openpose_train/training_results_gines_new/pose_video3/model/"

for i in [300000]:

    while 1:
        pfile = "pose_iter_%d.caffemodel" % i
        try:
            print(V100_MODEL+pfile)
            copyfile(V100_MODEL+pfile, HOME_MODEL + "/pose_iter_XXXXXX.caffemodel")
            break
        except:
            print("No File")
            time.sleep(1)

    #stop

    print(pfile)
    output = os.system("bash eval_posetrack.sh tracking BODY_25B")

    print output


