# READ ME

This document explains how to run multimodal medical predictions

Running landmarks ideas from landmark processing are taken from James McCouats paper/repo [here](https://github.com/jfm15/)


General Discription:
The dataloader inputs an image, and multiple heatmaps of the landmarks as the ground truth.


Output is a image of landmarks (heatmap). We then can extract the 'hotest' point for comparison.


## Models
Different models are explored in this repo, which have metadata added at different locations in the network. 


### Baseline
to add hrnet

> go into main dir and cd ..
git clone git@github.com:HRNet/HigherHRNet-Human-Pose-Estimation.git

One manual edit lib/models/pose_higher_hrnet.py
def get_pose_net(cfg, is_train, **kwargs):
    model = PoseHigherResolutionNet(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED, verbose=1)

    return model

you cant set verbose in your config as it would have to be the main variable

### Early Fusion Models

### Joint Fusion

### Late Fusion 
Adding metadata at the end of the model.