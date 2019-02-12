# **KeypointNet**

A Residual Neural Network for Extracting Feature Points of FSD Cones

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

What things you need to install the software and how to install them

1) Install Anaconda (recommend)
2) Install Pytorch, scikit-image
3) Install [sloth](https://github.com/wincle/sloth) for labeling image data of keypoints.

```bash
git clone https://github.com/chentairan/KeypointNet.git
```

### Run

A step by step series of examples that tell you how to get running



1) Put the cone images (*.jpg, 80 x 80) into `./dataset/` directory

2) Run the `generate.py` to  generate the `keypoints.json`

3) Run the `train.py`
