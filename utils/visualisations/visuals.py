import matplotlib.pyplot as plt
import os
class visuals():
    def __init__(self, save_path) -> None:
        self.save_path = save_path
        pass
    
    def channels_thresholded(self, output):
        #theshold then add all channels together
        for c in range(output.shape[0]):
            try:
                compressed_channels = compressed_channels+output[c]
            except:
                compressed_channels = output[c]
        return compressed_channels

    def heatmaps(self, image, output, target_points, predicted_points, w_landmarks=True):
        fig, ax = plt.subplots(1, 1)
        image = image.detach().cpu().numpy()
        output = output.detach().cpu().numpy()
        predicted_points = predicted_points.detach().cpu().numpy()
        target_points = target_points.cpu().detach().numpy()

        _output = self.channels_thresholded(output)
        ax.imshow(_output, cmap='inferno', alpha = 1)

        ax.axis('off')

        if w_landmarks == True:
            ax.imshow(image, cmap='Greys_r')
            #add landmarks
            ax.scatter(target_points[:, 0], target_points[:, 1], color='lime', s=10)
            ax.scatter(predicted_points[:, 0], predicted_points[:, 1], color='red', s=10)
        else:
            ax.imshow(image, cmap='Greys_r',alpha=0.4)
            
        plt.savefig(self.save_path)
        plt.close()

    