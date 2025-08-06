import sys
import os
import pathlib
import torch
target_path = pathlib.Path(os.path.abspath(__file__)).parents[1]
sys.path.append(target_path)
from ..evaluation_helper import evaluation_helper
import pandas as pd
import numpy as np
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
        if len(pred_map.shape) == 2:
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
                scaled_indices = torch.multiply(indices, pixel_size.to('cpu').float())

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

class landmark_overall_metrics():
    def __init__(self, pixelsize, unit='pixels') -> None:
        self.pixel_size = pixelsize.detach().cpu().numpy()
        self.unit=unit
        pass

    def get_sdr_statistics(self, radial_errors, thresholds):
        #if radial errors are a df convert to numpy
        if type(radial_errors) == pd.Series:
            radial_errors = radial_errors.to_numpy()
            
        successful_detection_rates = []
        for threshold in thresholds:
            if self.unit == 'mm':
                #convert radial erradial_errorsror to pixels
                radial_errors = radial_errors*self.pixel_size[0]
                filter = np.where(radial_errors < threshold, 1.0, 0.0)
            else:
                filter = np.where(radial_errors < threshold, 1.0, 0.0)

            sdr = 100 * np.sum(filter) / np.size(radial_errors)
            successful_detection_rates.append(sdr)
            
        txt = "Successful Detection Rates: "
        i = 0
        for sdr_rate in successful_detection_rates:
            #print(thresholds[i],thresholds[i]/self.pixel_size[0], sdr_rate)
            if self.unit == 'mm':
                txt += "{:.2f} mm [{:.2f}\t pixels]: {:.2f}%\t".format(thresholds[i], thresholds[i]/self.pixel_size[0], sdr_rate)
            else:
                txt += "{:.2f} pixels [{:.2f}\t mm]: {:.2f}%\t".format(thresholds[i], thresholds[i]*self.pixel_size[0], sdr_rate)
            
            i=i+1
    

        return successful_detection_rates, txt
