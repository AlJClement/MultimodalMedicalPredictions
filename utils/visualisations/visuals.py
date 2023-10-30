import matplotlib.pyplot as plt

class visuals():
    def __init__(self) -> None:
        pass

    def figure(self, image, graphics_function, args, save=True, save_path=""):
        fig, ax = plt.subplots(1, 1)

        #image = image.cpu().detach().numpy()
        ax.imshow(image[0], cmap='gray')

        graphics_function(ax, *args)

        ax.axis('off')
        plt.tight_layout()

        try:
            h, w = image[0].size()
        except:
            h, w = image[0].shape
        fig.set_size_inches(w / 100.0, h / 100.0)
        fig.set_dpi(100)

        if save:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def val_figure(self,image, output, predicted_points, target_points, save=True, save_path=""):
        self.figure(image, heatmaps_and_preds, (output, predicted_points, target_points), save=save, save_path=save_path)

