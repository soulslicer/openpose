## STAF Algorithm

This is a real-time live-demo of the STAF multi-person pose detection and tracker. Build instructions are similar to OpenPose as it is built of it's internal code.

`cd models; sh getModels.sh`

`build/examples/openpose/openpose.bin --model_pose BODY_21A --tracking 1  --render_pose 1`

### Limitations

As explained in the paper, one limitation is that this method is unable to handle scene changes for now. It will require refreshing the state or rerunning the algorithm for a new scene. Also, due to the smaller capacity of the network, tiny people in close proximity is not handled as well as a deeper network with more weights. This is something to explore in the future as well
