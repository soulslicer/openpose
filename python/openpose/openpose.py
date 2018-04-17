import numpy as np
import ctypes as ct
import cv2
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

class OpenPose(object):
    _libop= np.ctypeslib.load_library('_openpose', dir_path+'/_openpose.so')
    _libop.newOP.argtypes = [
        ct.c_int, ct.c_char_p, ct.c_char_p, ct.c_char_p, ct.c_float, ct.c_float, ct.c_int, ct.c_float, ct.c_int, ct.c_bool, ct.c_char_p]
    _libop.newOP.restype = ct.c_void_p
    _libop.delOP.argtypes = [ct.c_void_p]
    _libop.delOP.restype = None

    _libop.forward.argtypes = [
        ct.c_void_p, np.ctypeslib.ndpointer(dtype=np.uint8),
        ct.c_size_t, ct.c_size_t,
        np.ctypeslib.ndpointer(dtype=np.int32), np.ctypeslib.ndpointer(dtype=np.uint8), ct.c_bool]
    _libop.forward.restype = None

    _libop.getOutputs.argtypes = [
        ct.c_void_p, np.ctypeslib.ndpointer(dtype=np.float32)]
    _libop.getOutputs.restype = None

    _libop.poseFromHeatmap.argtypes = [
        ct.c_void_p, np.ctypeslib.ndpointer(dtype=np.uint8),
        ct.c_size_t, ct.c_size_t,
        np.ctypeslib.ndpointer(dtype=np.uint8),
        np.ctypeslib.ndpointer(dtype=np.float32), np.ctypeslib.ndpointer(dtype=np.int32)]
    _libop.poseFromHeatmap.restype = None

    def __init__(self, params):
        self.op = self._libop.newOP(params["logging_level"],
                                    params["output_resolution"],
                                    params["net_resolution"],
                                    params["model_pose"],
                                    params["alpha_pose"],
                                    params["scale_gap"],
                                    params["scale_number"],
                                    params["render_threshold"],
                                    params["num_gpu_start"],
                                    params["disable_blending"],
                                    params["default_model_folder"])

    def __del__(self):
        self._libop.delOP(self.op)

    def forward(self, image, display = False):
        shape = image.shape
        displayImage = np.zeros(shape=(image.shape),dtype=np.uint8)
        size = np.zeros(shape=(3),dtype=np.int32)
        self._libop.forward(self.op, image, shape[0], shape[1], size, displayImage, display)
        array = np.zeros(shape=(size),dtype=np.float32)
        self._libop.getOutputs(self.op, array)
        if display:
            return array, displayImage
        return array

    def poseFromHM(self, image, hm):
        shape = image.shape
        displayImage = np.zeros(shape=(image.shape),dtype=np.uint8)
        size = np.zeros(shape=(4),dtype=np.int32)
        size[0] = hm.shape[0]
        size[1] = hm.shape[1]
        size[2] = hm.shape[2]
        size[3] = hm.shape[3]
        self._libop.poseFromHeatmap(self.op, image, shape[0], shape[1], displayImage, hm, size)
        return displayImage

if __name__ == "__main__":
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
    params["default_model_folder"] = dir_path + "/../../models/"
    openpose = OpenPose(params)
