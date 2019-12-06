<<<<<<< HEAD
## STAF Algorithm

This is a real-time live-demo of the STAF multi-person pose detection and tracker. Build instructions are similar to OpenPose as it is built of it's internal code.

`cd models; sh getModels.sh`

`build/examples/openpose/openpose.bin --model_pose BODY_21A --tracking 1  --render_pose 1`
=======
# OpenPose STAF GoVertical

This is a real-time live version of the STAF multi-person pose detection and tracker.

## Get Started

How to run a video through openpose-staf

#### Currently tested and ran on a Google Cloud Instance:

Region: US-West-1b
Machine Configuration: n1-standard-8 (8vCPUs, 32gb memory)
GPU: 1 Nvidia Tesla K80
OS Image: Deep Learning Image: TensorFlow 1.15.0 m41 (TensorFlow 1.15.0 with CUDA 10.0 and Intel® MKL-DNN, Intel® MKL.)
Disk Size: 500GB
Note: if you want a jupyter server, you must allow http/https traffic & have a static IP

```
# Launch instance, then pull Docker
docker pull cwaffles/openpose
# get inside container
docker run -it --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 cwaffles/openpose

# update openpose on docker container
git fetch origin
git pull
# Add remote repo
git remote add brody-op https://github.com/BroderickHigby/openpose.git
git fetch brody-op
git checkout --track brody-op/staf

# Build C code
cd build/; make -j `nproc`; cd ../
## Note: this may actually get the models as part of the build process as well (so we may not need to run getModels.sh after)

# get models
cd models; sh getModels.sh

# Run and Save .mp4 files
apt-get update
apt-get install ffmpeg

# Edit code in GitHub and push to repo, if needed
```

## Run Inference

Upload video to `./video` folder via GitHub, Docker connect, or wget. Example videos already in the repo.

For help on flags, type: `build/examples/openpose/openpose.bin --help`

### For local machines

`build/examples/openpose/openpose.bin --model_pose BODY_25 --tracking 1 --render_pose 1 --write_video <VideoName> --write_json output/ --video <VideoName>`

### For servers

`build/examples/openpose/openpose.bin --model_pose BODY_25 --tracking 1 --render_pose 1 --write_video <VideoName> --write_json output/ --video <VideoName> --display 0`

i.e.
`./build/examples/openpose/openpose.bin --video video/ufc.mp4 --write_video output/ufc.mp4 --write_json output/ --model_pose BODY_25 --render_pose 2 --tracking 0 --number_people_max 1 --display 0`

### Save Docker container changes

`sudo docker commit CONTAINER_ID nginx-template`

### Recursively Save video from output folder

`tar -czvf staf-json.tar.gz output`
>>>>>>> b6d7a494cdbf6934d3d66b3bf7d79db0768a0197

### Limitations

As explained in the paper, one limitation is that this method is unable to handle scene changes for now. It will require refreshing the state or rerunning the algorithm for a new scene. Also, due to the smaller capacity of the network, tiny people in close proximity is not handled as well as a deeper network with more weights. This is something to explore in the future as well
