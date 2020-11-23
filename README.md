# Rotation RetinaNet

The codes build R-RetinaNet for arbitrary-oriented object detection. It supports the following datasets: DOTA, HRSC2016, ICDAR2013, ICDAR2015, UCAS-AOD, NWPU VHR-10, VOC2007. 

### Performance
This implementation reaches 24 fps on RTX 2080 Ti. 

The performance is shown as follow:

#### HRSC2016

Note that VOC07 metric is used for evaluation.

| Dataset          | Bbox |Backbone   | Input size | mAP       |
| ---------------  | ---- | ---------- | --------- |---------|
| HRSC2016         | OBB  | ResNet-50  | 416 x 416  | 80.81    |
| UCAS-AOD         | OBB  |ResNet-50  | 800 x 800  | 87.57     |
| ICDAR 2013       | OBB  |ResNet-50 | 800 x 800  | 77.20    |
| ICDAR 2015       | HBB  |ResNet-50 | 800 x 800  | 77.50 |
| NWPU VHR-10      | HBB  |ResNet-50 | 800 x 800  | 86.40 |

Note that VOC07 metric is used for HRSC2016, UCAS-AOD, NWPU VHR-10, F1 score for IC13 and IC13. 3 anchors are preset for each location in feature maps. All experiments are conducted with data augmentation including random flip, rotation and HSV color space transform.



## Getting Started
### Installation
Build the Cython  and CUDA modules:
```
cd $ROOT/utils
sh make.sh
```

Install DotaDevKit:

```
sudo apt-get install swig
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```

### Inference

```
python demo.py
```

### Train
1. prepare dataset and move it into the `$ROOT` directory.
2. generate imageset files:
```
cd $ROOT/datasets
python generate_imageset.py
```
3. start training:
```
python train.py
```
### Evaluation
prepare labels, take hrsc for example:
```
cd $ROOT/datasets/evaluate
python hrsc2gt.py
```
start evaluation:
```
python eval.py
```
Note that :

- the script  needs to be executed **only once**.
- the imageset file used in `hrsc2gt.py` is generated from `generate_imageset.py`.

## Detection Result

<img src="outputs\100001410.jpg" alt="100001410"  />