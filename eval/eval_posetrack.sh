#!/bin/bash
# Script to extract COCO JSON file for each trained model
#clear && clear

rm -rf posetrack_results
mkdir posetrack_results
mkdir posetrack_results/op_output

OPENPOSE_FOLDER=$(pwd)/../
POSETRACK_FOLDER=$(pwd)/posetrack/images/val

# for file in $POSETRACK_FOLDER/* ; do 
#   echo "$file"
# done

# Not coded for Multi GPU Yet

N=1
(
for folder in $POSETRACK_FOLDER/* ; do 
  if [[ -d "$folder" && ! -L "$folder" ]]; then
    ((i=i%N)); ((i++==0)) && wait
    process=$((i%N));

    filename="${folder##*/}"

    # Operation
    cd $OPENPOSE_FOLDER;
    ./build/examples/openpose/openpose.bin \
        --model_pose BODY_25B \
        --image_dir $folder \
        --write_json eval/posetrack_results/op_output/$filename \
        --render_pose 0 --display 0 &

    # sleep 1 &
    # echo "$folder $var is a directory" & 

  fi; 
done
)
