# READ ME

This document explains how to run multimodal medical predictions

Running landmarks ideas from landmark processing are taken from James McCouats paper/repo [here](https://github.com/jfm15/)


General Discription:
The dataloader inputs an image, and multiple heatmaps of the landmarks as the ground truth.


Output is a image of landmarks (heatmap). We then can extract the 'hotest' point for comparison.


## Models
Different models are explored in this repo, which have metadata added at different locations in the network. 
### Early Fusion Models

### Joint Fusion

### Late Fusion 
Adding metadata at the end of the model.