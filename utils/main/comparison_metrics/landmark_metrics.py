import sys
import os
import pathlib
import torch
target_path = pathlib.Path(os.path.abspath(__file__)).parents[1]
sys.path.append(target_path)
from ..evaluation_helper import evaluation_helper
class landmark_metrics():
    def __init__(self) -> None:
        '''calcualtions between two different sets of landmarks'''
        pass

    def get_radial_errors(self, pred_points, pred_map, targ_points, true_map, pixelsize, mean=False):
        if len(pred_points.shape) == 2:
            pred_points = pred_points.unsqueeze(dim=0)
            targ_points = targ_points.unsqueeze(dim=0)
        displacement = pred_points - targ_points
        per_landmark_error = torch.norm(displacement.float(), dim=2)
        landmark_error_ls = []
        _tmp = per_landmark_error[0].detach().cpu().numpy()
        i = 1
        for l in _tmp:
            landmark_error_ls.append(['landmark radial error p'+str(i) , l])
            i = i +1
        return landmark_error_ls
        

    def get_eres(self, pred, pred_map, true, true_map, pixelsize):        
        # Expected radial error, can handle multiple inputs or one.
        # explains how uncertian a model is based on heatmap output and predicted landmark
        # if pred output is dim x,y add channel for threshold to work
        if len(pred.shape) == 2:
            pred = pred.unsqueeze(dim=0)
            pred_map = pred_map.unsqueeze(dim=0)
        pred_thresholded = evaluation_helper().get_thresholded_heatmap(pred_map, pred)
        
        eres_per_image = []
        for pred_thresholded, predicted_points, pixel_size in zip(pred_thresholded, pred, pixelsize):
            ere_per_heatmap = []
            for pred_thresh, predicted_point in zip(pred_thresholded, predicted_points):
               
                indices = torch.nonzero(pred_thresh)
                pred_flattened = torch.flatten(pred_thresh)
                flattened_indices = torch.nonzero(pred_flattened)
                significant_values = pred_flattened[flattened_indices]
                scaled_indices = torch.multiply(indices, pixel_size.float())

                displacement_vectors = torch.sub(scaled_indices, predicted_point)
                distances = torch.norm(displacement_vectors, dim=1)
                ere = torch.sum(torch.multiply(torch.squeeze(significant_values), distances))
                ere_per_heatmap.append(ere)
            eres_per_image.append(torch.stack(ere_per_heatmap))        

            ere_ls =[]
            _ere = eres_per_image[0].detach().cpu().numpy()
            i = 1
            for p in _ere:
                ere_ls.append(['ere p'+str(i) , p])
                i = i +1
            return ere_ls
