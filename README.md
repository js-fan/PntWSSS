# Toward Practical Weakly Supervised Semantic Segmentation via Point-Level Supervision

This is the implementation of the method described in paper: toward practical weakly supervised semantic segmentation via point-level supervision.

### Requirements
- Python3.7+
- Pytorch1.0+
- Numpy, OpenCV
- Pydensecrf

### Usage
1. Prepare the dataset, e.g., [VOC](http://host.robots.ox.ac.uk/pascal/VOC/), [Cityscapes](https://www.cityscapes-dataset.com/), and [ADE20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/). We only use the image and point-level labels as supervision, where the used point labels can be downloaded [here](https://drive.google.com/drive/folders/1OhgcQNcX1RfwzS0VyYhVoPsPUBzv88Ec?usp=sharing).

2. Run the scripts to reproduce the results. Change the dataset pathes if necessary, which are typically determined by `train-image-root`, `test-image-root`, `train-label-file`, and `test-gt-root`. The results with different datasets can be reproduced by simply running scripts train_xxx.py, e.g.,
```bash
python train_voc_all.py --gpus 0,1,2,3
```
The scripts will automatically conduct evaluation and print logs during running.

