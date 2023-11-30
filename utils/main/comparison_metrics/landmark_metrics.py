import torch



class landmark_metrics():
    def __init__(self) -> None:
        '''calcualtions between two different sets of landmarks'''
        pass

    def get_eres(self, pred, predicted_points_scaled, pixel_sizes):        
        # flip points
        pred_thresholded = self.get_thresholded_heatmap(pred, predicted_points_scaled)
        
        eres_per_image = []
        for heatmap_stack, predicted_points_per_image, pixel_size in zip(pred_thresholded, predicted_points_scaled, pixel_sizes):
            ere_per_heatmap = []
            for pdf, predicted_point in zip(heatmap_stack, predicted_points_per_image):
                indices = torch.nonzero(pdf)
                pdf_flattened = torch.flatten(pdf)
                flattened_indices = torch.nonzero(pdf_flattened)
                significant_values = pdf_flattened[flattened_indices]

                scaled_indices = torch.multiply(indices, pixel_size.float())
                displacement_vectors = torch.sub(scaled_indices, predicted_point)
                distances = torch.norm(displacement_vectors, dim=1)
                ere = torch.sum(torch.multiply(torch.squeeze(significant_values), distances))
                ere_per_heatmap.append(ere)
            eres_per_image.append(torch.stack(ere_per_heatmap))

        return torch.stack(eres_per_image)