#!/bin/bash
# Script to extract COCO JSON file for each trained model
clear && clear

mkdir testing_results

if [ ! -d "val2017" ]; then
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
rm val2017.zip
fi

if [ ! -d "annotations" ]; then
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
rm annotations_trainval2017.zip
fi

if [ ! -d "cocoapi-master" ]; then
wget https://github.com/cocodataset/cocoapi/archive/master.zip
mv master.zip coco_api.zip
unzip coco_api.zip
rm coco_api.zip
fi

cd cocoapi-master/PythonAPI
make

