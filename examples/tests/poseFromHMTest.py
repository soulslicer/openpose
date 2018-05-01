import os
os.environ["GLOG_minloglevel"] = "1"
import caffe
import cv2
import numpy as np
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('../../python')
dir_path + "/../../models/"
from openpose import OpenPose

# Params
class Param:
    caffemodel = dir_path + "/../../../models/pose/coco/pose_iter_440000.caffemodel"
    prototxt = dir_path + "/../../../models/pose/coco/pose_deploy_linevec.prototxt"
    boxsize = 368
    padValue = 0

# Load net
params = dict()
params["logging_level"] = 3
params["output_resolution"] = "-1x-1"
params["net_resolution"] = "-1x368"
params["model_pose"] = "COCO"
params["alpha_pose"] = 0.6
params["scale_gap"] = 0.3
params["scale_number"] = 1
params["render_threshold"] = 0.05
params["num_gpu_start"] = 0
params["disable_blending"] = False
params["default_model_folder"] = dir_path + "/../../../models/"
openpose = OpenPose(params)
caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net(Param.prototxt, Param.caffemodel, caffe.TEST)
print "Net loaded"

def show_image(img):
    if img is not None:
        cv2.imshow("win",img)
        key = cv2.waitKey(15)
        if key == 32:
            while 1:
                key = cv2.waitKey(15)
                if key == 32:
                    break
        return key

def pad_image(image, padValue, bbox):
    h = image.shape[0]
    h = min(bbox[0], h);
    w = image.shape[1]
    bbox[0] = (np.ceil(bbox[0]/8.))*8;
    bbox[1] = max(bbox[1], w);
    bbox[1] = (np.ceil(bbox[1]/8.))*8;
    pad = np.zeros(shape=(4))
    pad[0] = 0;
    pad[1] = 0;
    pad[2] = int(bbox[0]-h);
    pad[3] = int(bbox[1]-w);
    imagePadded = image
    padDown = np.tile(imagePadded[imagePadded.shape[0]-2:imagePadded.shape[0]-1,:,:], [int(pad[2]), 1, 1])*0
    imagePadded = np.vstack((imagePadded,padDown))
    padRight = np.tile(imagePadded[:,imagePadded.shape[1]-2:imagePadded.shape[1]-1,:], [1, int(pad[3]), 1])*0 + padValue
    imagePadded = np.hstack((imagePadded,padRight))
    return imagePadded, pad

def unpad_image(image, padding):
    if padding[0] < 0:
        print "NOT IMPLEMENTED"
    elif padding[0] > 0:
        print "NOT IMPLEMENTED"

    if padding[1] < 0:
        print "NOT IMPLEMENTED"
    elif padding[1] > 0:
        print "NOT IMPLEMENTED"

    if padding[2] < 0:
        print "NOT IMPLEMENTED"
    elif padding[2] > 0:
        image = image[0:image.shape[0]-int(padding[2]),:]

    if padding[3] < 0:
        print "NOT IMPLEMENTED"
    elif padding[3] > 0:
        image = image[:,0:image.shape[1]-int(padding[3])]

    return image

def power_law(img, power):
    img=img.astype('float32')
    img /=255.0
    img = cv2.GaussianBlur(img,(5,5),5)
    img = img**power
    img *= (255.0)
    img = img.astype('uint8')
    return img

currIndex = 0
first_run = True
def func(frame):
    # Reshape
    height, width, channels = frame.shape
    scaleImage = float(Param.boxsize) / float(height)
    rframe = cv2.resize(frame, (0,0), fx=scaleImage, fy=scaleImage)
    bbox = [Param.boxsize, max(rframe.shape[1], Param.boxsize)];
    imageForNet, padding = pad_image(rframe, Param.padValue, bbox)
    imageForNet = imageForNet.astype(float)
    imageForNet = imageForNet/256. - 0.5
    imageForNet = np.transpose(imageForNet, (2,0,1))
    #print imageForNet.shape

    global first_run
    if first_run:
        in_shape = net.blobs['image'].data.shape
        in_shape = (1, 3, imageForNet.shape[1], imageForNet.shape[2])
        net.blobs['image'].reshape(*in_shape)
        net.reshape()
        first_run = False
        print "Reshaped"

    net.blobs['image'].data[0,:,:,:] = imageForNet
    net.forward()
    heatmaps = net.blobs['net_output'].data[:,:,:,:]
    print heatmaps.shape

    # Pose from HM Test
    array, frame = openpose.poseFromHM(frame, heatmaps)
    print array
    show_image(frame)

    return frame


# Run Video
cap = cv2.VideoCapture(dir_path + "/../../../examples/media/trump.mp4")
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Func
    frame = func(frame)

    # Display the resulting frame
    # show_image(frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
