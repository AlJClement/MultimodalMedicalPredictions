import torch

class evaluation():
    def __init__(self) -> None:
        return

    def get_hottest_points(self, output_stack):
        ''' #JAMES MCCOUAT CODE
        # Get the predicted landmark point from the "hottest point"
        # Heatmap is a tensor of size (B, N, W, H) 
        '''
        _, _, w, h = output_stack.size()
        flattened_heatmaps = torch.flatten(output_stack, start_dim=2)
        hottest_idx = torch.argmax(flattened_heatmaps, dim=2)
        x = torch.div(hottest_idx, h, rounding_mode="floor")
        y = torch.remainder(hottest_idx, h)
        return torch.stack((y, x), dim=2)
    
    def get_thresholded_heatmap(self, output, significant_radius=0.05):
        '''This function takes the output channels and thresholds the heatmaps to the significant radius, then renormalizes'''
        flattened_heatmaps = torch.flatten(output, start_dim=2)
        max_per_heatmap, _ = torch.max(flattened_heatmaps, dim=2, keepdim=True)
        max_per_heatmap = torch.unsqueeze(max_per_heatmap, dim=3)
        normalized_heatmaps = torch.div(output, max_per_heatmap)
        
        zero_tensor = torch.tensor(0.0).cuda() if output.is_cuda else torch.tensor(0.0)
        filtered_heatmaps = torch.where(normalized_heatmaps > significant_radius, normalized_heatmaps,
                                        zero_tensor)
        flattened_filtered_heatmaps = torch.flatten(filtered_heatmaps, start_dim=2)
        sum_per_heatmap = torch.sum(flattened_filtered_heatmaps, dim=2, keepdim=True)
        sum_per_heatmap = torch.unsqueeze(sum_per_heatmap, dim=3)
        thresholded_output = torch.div(filtered_heatmaps, sum_per_heatmap)

        return thresholded_output
        
    def get_eres(self, output, predicted_points_scaled, pixel_sizes):        
        # flip points
        # predicted_points_scaled = torch.flip(predicted_points_scaled, dims=[2])
        # 
        output_thresholded = self.get_thresholded_heatmap(output)
        
        eres_per_image = []
        for heatmap_stack, predicted_points_per_image, pixel_size in zip(output_thresholded, predicted_points_scaled, pixel_sizes):
            ere_per_heatmap = []
            for pdf, predicted_point in zip(heatmap_stack, predicted_points_per_image):
                indices = torch.nonzero(pdf)

                pdf_flattened = torch.flatten(pdf)
                flattened_indices = torch.nonzero(pdf_flattened)
                significant_values = pdf_flattened[flattened_indices]

                scaled_indices = torch.multiply(indices, pixel_size)
                displacement_vectors = torch.sub(scaled_indices, predicted_point)
                distances = torch.norm(displacement_vectors, dim=1)
                ere = torch.sum(torch.multiply(torch.squeeze(significant_values), distances))
                ere_per_heatmap.append(ere)
            eres_per_image.append(torch.stack(ere_per_heatmap))

        return torch.stack(eres_per_image)
    
    def get_landmarks(self, output, target_points, pixels_sizes):
        # Evaluate radial error
        # Predicted points has shape (B, N, 2)
        predicted_points = self.get_hottest_points(output)
        scaled_predicted_points = torch.multiply(predicted_points, pixels_sizes)

        # Get landmark center (B, N, 2)
        target_points = self.get_hottest_points(target_points) #should be center of gaussian
        scaled_target_points = torch.multiply(predicted_points, pixels_sizes)

        # Get expected radial error scores
        eres = self.get_eres(scaled_predicted_points, scaled_target_points)

        # return scaled_predicted_points, scaled_target_points, eres
        return scaled_predicted_points, scaled_target_points, eres
