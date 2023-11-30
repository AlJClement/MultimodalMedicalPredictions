import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd

class comparison():
    def __init__(self) -> None:
        pass        

    def confusion_matrix_multiclass(self, classes, confusion_matrix_multiclasses):
        fig, ax= plt.subplots(1, confusion_matrix_multiclasses.shape[0])
        fig.set_figheight(2)
        fig.set_figwidth(20)
        for c in range(confusion_matrix_multiclasses.shape[0]):
            class_name = classes[c]
            cm = confusion_matrix_multiclasses[c]
            sns.heatmap(cm, annot=True, fmt='g', ax=ax[c], vmax= len(self.gt_class_arr)) 
            # labels, title and ticks
            ax[c].set_xlabel('Predicted labels')
            ax[c].set_ylabel('True labels')
            ax[c].set_title('Confusion Matrix: '+ class_name)
        
        plt.savefig('./output/Confusion_Matrix_allclasses')

    def to_one_hot(y, num_classes):
        y = y.squeeze()
        store = np.eye(num_classes)[y]
        return store
    #T-SNE
    def get_tsne_map(self, pred_diagnosis, true_diagnosis):

        x = self.to_one_hot(x, pred_diagnosis.is_unique())

        tsne = TSNE(n_components=2, verbose=1, random_state=123)
        z = tsne.fit_transform(x) 
        df = pd.DataFrame()
        df["y"] = true_diagnosis
        df["comp-1"] = z[:,0]
        df["comp-2"] = z[:,1]

        sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                        palette=sns.color_palette("hls", 5),
                        data=df).set(title="Class Predictions") 
        
        fig = sns.get_figure()
        fig.savefig("./output/tsne.png")