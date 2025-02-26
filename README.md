# Bridging Inverse Depth and Metric Depth through Prior Knowledge-based Self-Supervised Depth Completion

### 1.Pre-Execution Setup  

Before running the program:  
1. Download the official [DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2) library and [Checkpoints](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth#pre-trained-models).  
2. Place them in the `Process_est/ `  
3. Rename the DepthAnythingv2 folder to `da2`.  

### 2.Dataset Processing Instructions

You can easily download and process the following datasets:

1. [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion)  
2. [NYUV2 Dataset-H5](http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz)
3. [VOID Dataset](https://drive.google.com/open?id=1rzTFD35OCxMIguxLDcBxuIdhh5T2G7h4)  

To process the data, run the following scripts:  
- `Process_est/kitti.py`  
- `Process_est/nyuv2.py`
- `Process_est/void-relative.py`  
- `Process_est/nyu-relative.py`
- For KITTI: We have processed the train, val, saval and test of the KITTI dataset, you can also download it directly [Google Driver](https://drive.google.com/drive/folders/1MM3gwcAIz6JAED2PLrvwGRLGO8yYT5l5?usp=sharing).
- For NYUv2：We provide two scripts that can directly insert the processed data into the h5 file with keys('est','est_r'). You can do this easily.
- For VOID：We provide a script that can directly insert the processed data into raw datasets. You can do this easily.



