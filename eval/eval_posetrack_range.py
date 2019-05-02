import os

for i in range(400, 1600, 200):
    command = "cd /home/raaj/openpose_staf/models/pose/body_25b_video; python copy.py pose_iter_%d.caffemodel" % i
    output = os.system(command)

    print("pose_iter_%d.caffemodel" % i)
    output = os.system("bash eval_posetrack.sh  tracking")

    print output