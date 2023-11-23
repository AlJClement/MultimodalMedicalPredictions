import torch

class evaluation_helper():
    def __init__(self) -> None:
        '''adapted from james mcouat script
        helper evaluation functions for getting hotest point, get landmarks from the output, thresholding the heat maps etc.'''
        return

    def get_hottest_points(self, img):
        '''
        # Get the predicted landmark point from the "hottest point" in each channel
        # img tensor of size (B, C, W, H), where C is the channel
        '''
        B, C, W, H = img.size()
        #flatten w/h so tensor is (B,C,WxH)
        flattened_heatmaps = torch.flatten(img, start_dim=2)
        #get index of hotest point in each channel (B, C)
        hottest_idx = torch.argmax(flattened_heatmaps, dim=2)

        #get x,y coords for these hotest points
        x = torch.div(hottest_idx, H, rounding_mode="floor")
        y = torch.remainder(hottest_idx, H)

        return torch.stack((y, x), dim=2)
    
    def get_thresholded_heatmap(self, pred, predicted_points_scaled, significant_radius=0.05):
        '''This function takes the output channels and thresholds the heatmaps to the significant radius, then renormalizes those values'''
        #dimension depend on the size of the pred. which depends on the number of annotators/if we are combining annotators to one map
        #flip points
        predicted_points_scaled = torch.flip(predicted_points_scaled, dims=[2])
        #print('out stack size', pred.shape)
        flattened_heatmaps = torch.flatten(pred, start_dim=2)
        #print('flattened_heatmaps size:', flattened_heatmaps.shape)
        max_per_heatmap, _ = torch.max(flattened_heatmaps, dim=2, keepdim=True)
        max_per_heatmap = torch.unsqueeze(max_per_heatmap, dim=3)
        #print(' max_per_heatmap size', max_per_heatmap.shape)

        normalized_heatmaps = torch.div(pred, max_per_heatmap)

        zero_tensor = torch.tensor(0.0).cuda() if pred.is_cuda else torch.tensor(0.0)
        filtered_heatmaps = torch.where(normalized_heatmaps > significant_radius, normalized_heatmaps,
                                        zero_tensor)
        flattened_filtered_heatmaps = torch.flatten(filtered_heatmaps, start_dim=2)
        sum_per_heatmap = torch.sum(flattened_filtered_heatmaps, dim=2, keepdim=True)
        sum_per_heatmap = torch.unsqueeze(sum_per_heatmap, dim=3)
        thresholded_output = torch.div(filtered_heatmaps, sum_per_heatmap)

        return thresholded_output
    
    def get_landmarks(self, pred, target_points, pixels_sizes):
        # Evaluate radial error
        # Predicted points has shape (B, N, 2)
        predicted_points = self.get_hottest_points(pred)
        scaled_predicted_points = torch.multiply(predicted_points, pixels_sizes)

        # Get landmark center (B, N, 2)
        target_points = self.get_hottest_points(target_points) #should be center of gaussian
        scaled_target_points = torch.multiply(target_points, pixels_sizes)

        return  scaled_target_points[0], scaled_predicted_points
