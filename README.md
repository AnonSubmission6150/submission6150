Note: Contributors appeared in this page all come from the original mmsegmentation repository.

## Train
DeepGlobe
```
./tools/dist_train_isdnet.sh configs/isdnet/isdnet_1224x1224_80k_deepglobe.py 4
```
InriaAerial
```
./tools/dist_train_isdnet.sh configs/isdnet/isdnet_2500x2500_40k_InriaAerial.py 4
```
Cityscapes
```
./tools/dist_train_isdnet.sh configs/isdnet/isdnet_512x512_160k_cityscpaes.py 8
```
## Inference
```
python tools/test.py config_file checkpoints_file --eval mIoU
```
## FPS test
```
python tools/fps_test.py config_file --h height of the test image --w width of the test image
```
## Installation
Our code is based on mmsegmentation (version 0.16.0), you should install this before runing.
The install instructions can refer to the original mmsegmentation codebase. 
