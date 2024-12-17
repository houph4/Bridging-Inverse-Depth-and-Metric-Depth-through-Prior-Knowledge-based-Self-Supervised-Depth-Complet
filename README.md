# Depth-Completion-based-on-foundation-model

### Pre-Execution Setup  

Before running the program:  
1. Download the official [DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2) library and [Checkpoints](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth#pre-trained-models).  
2. Place them in the `Process_est/ `  
3. Rename the folder to `da2`.  

### Dataset Processing Instructions

You can easily process the following datasets:

1. [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion)  
2. [NYUV2 Dataset-H5](http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz)  

To process the data, run the following scripts:  
- `Process_est/kitti.py`  
- `Process_est/nyuv2.py`
- For KITTI:We have processed the train, val, saval and test of the KITTI dataset, you can also download it directly [Google Driver](https://drive.google.com/drive/folders/1MM3gwcAIz6JAED2PLrvwGRLGO8yYT5l5?usp=sharing)
- For NYUv2：We provide a script that can directly insert the processed data into the h5 file. You can do this easily.



