# Prepare Matterport3D Data

1. Download Matterport3D data [HERE](https://niessner.github.io/Matterport/). Only `region_segmentations` is needed.

2. Unzip the dataset and move `organize_as_scannet.py` into the folder. The file directory should be like:

```.
...
├── organize_as_scannet.py
└── v1
    └── scans
        ├── ...
        ├── ZMojNkEp431
        └── zsNo4HB9uLZ
            └── region_segmentations
                ├── ...
                ├── resionx.fsegs.json
                ├── resionx.ply
                ├── resionx.semseg.json
                └── resionx.vsegs.json
```

3. In `region_segmentations`, the index x must be continuous (start from 0). Some folders do not comply with this rule and we manually changed the index.

4. Run `python organize_as_scannet.py`. Move/link the generated `for_scannet/scans` folder such that under `scans` there should be folders with names such as `scene0001_01`.

5. Extract point clouds and annotations (semantic seg, instance seg etc.) by running `python batch_load_matterport_data.py`, which will create a folder named `matterport_train_detection_data_md40` here.
