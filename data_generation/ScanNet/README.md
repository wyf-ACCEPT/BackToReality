# 1. Shape Processing

## 1.1. Download ModelNet40

The ModelNet40 dataset can be downloaded from [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip).  

Then create a soft link by running:

```.
ln -s /path/to/download/modelnet40_normal_resampled /path/to/this/directory/
```

## 1.2. Process the shapes

Run command:

```.
python modelnet40_tools.py
```

# 2. Virtual Scene Generation (Point-version)  

## 2.1. Download ScanNetV2

The ScanNetV2 dataset can be downloaded from [here](http://www.scan-net.org/).

## 2.2. Generate augmentation information

Run commands below:

```.
cd /path/to/BackToReality/detection/Votenet/scannet

python scannet_detection_dataset.py

mv scans_toadd_scarse.npy /path/to/BackToReality/data_generation/ScanNet/CONFIG
```

## 2.3. Generate the positions for shapes

Run commands below:

```.
cd /path/to/BackToReality/data_generation/ScanNet

python scannet_scene_synthesis.py

mv augment_random_positions_scannet /path/to/BackToReality/detection/Votenet/scannet
```

## 2.4. Use the positions to generate virtual scenes

Run commands below:

```.
cd /path/to/BackToReality/detection/Votenet/scannet

python batch_load_scannet_data_virtual.py
```
