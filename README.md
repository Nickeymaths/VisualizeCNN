# VisualizingCNN

A PyTorch implementation of the 2014 ECCV paper "Visualizing and understanding convolutional networks"


## Usage
Each experiment writed to different .py file
### Run visualization experiment

```bash
python visualize_exp.py
```
### Run occlude sensitative experiment

```bash
python occlude_exp.py
```
### Run stable experiment

```bash
python stable_exp.py
```

## Requirement

```bash
Pytorch == 0.4.0
opencv-python == 3.4.0.12
```

## Notes

The network use Alexnet pretrained from torchvision.models, the reconstruction proposal is human's labeling, rather model generate.

## Code reference
https://github.com/huybery/VisualizingCNN

Except files `vgg16_conv.py`, `vgg16_deconv.py` and some functions in `utils.py`, we implemented all rest of this project.
Deep dream: 
  https://github.com/gordicaleksa/pytorch-deepdream
  use all code and pre-train model of Google engineer.

## Member
| Member          | MSSV     | Contribution                                                                  |
|-----------------|----------|-------------------------------------------------------------------------------|
| Pham Thanh Vinh | 19021396 | Implement Alexnet conv, deconv, occlusion, visualization, stable experiments  |
| Nguyen Vu Hieu  | 19021276 | Correspondence experiment                                                     |
| Tran Ngoc Huong | 19021297 | Run deepdream experiment for visualizing                                      |
