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

The network use vgg16 pretrained from torchvision.models, the reconstruction proposal is human's labeling, rather model generate.