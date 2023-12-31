import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd

class comparison():
    def __init__(self,dataset_name):
        self.dataset_name = dataset_name

        if self.dataset_name == 'ddh':
            #threshold for classes at 60 and 43
            self.threshold_list = [60, 43]
        else:
            return ValueError('must set the threshold_dic dicitonary')
        pass        

    def confusion_matrix_multiclass(self, classes, confusion_matrix_multiclasses, loc='test'):
        fig, ax= plt.subplots(1, confusion_matrix_multiclasses.shape[0])
        fig.set_figheight(2)
        fig.set_figwidth(20)
        for c in range(confusion_matrix_multiclasses.shape[0]):
            class_name = classes[c]
            cm = confusion_matrix_multiclasses[c]
            sns.heatmap(cm, annot=True, fmt='g', ax=ax[c]) 
            # labels, title and ticks
            ax[c].set_xlabel('Predicted labels')
            ax[c].set_ylabel('True labels')
            ax[c].set_title('Confusion Matrix: '+ class_name)
        
        plt.savefig('./output/'+loc+'/Confusion_Matrix_allclasses.png')

    def to_one_hot(self, y, num_classes):
        y = y.squeeze().astype(int)
        hot = np.eye(num_classes)[y]
        return hot

    def true_vs_pred_scatter(self, pred, true, loc='test'):
        #plot predicted vs true value outputs.
        #order by true
        plt.clf()
        dataset = pd.DataFrame({'pred': pred, 'true': true}, columns=['pred', 'true'])
        dataset = dataset.sort_values('true')
        dataset.reset_index(drop=True)
        #plot true and pred
        patient = range(len(dataset))
        plt.scatter(patient,dataset['true'], c='g', alpha=1)
        plt.scatter(patient,dataset['pred'], c='r', alpha=0.5)
        plt.xlabel('Patient')
        plt.ylabel('Angle')

        for thresh in self.threshold_list:
            plt.axhline(y=thresh, color='b', linestyle='--')

        plt.savefig('./output/'+loc+'/true_vs_pred.png')
        return
    


    #T-SNE         visualisations.comparison().get_tsne_map(comparison_df['class pred'].to_numpy(),comparison_df['class true'].to_numpy())

    def get_tsne_map(self, pred_diagnosis, true_diagnosis):
        #convert from strings to values
        pred_diagnosis =np.unique(pred_diagnosis, return_inverse=True)[1]
        true_diagnosis =np.unique(true_diagnosis, return_inverse=True)[1]

        x = self.to_one_hot(pred_diagnosis, len(np.unique(true_diagnosis)))

        tsne = TSNE(n_components=2, verbose=1, random_state=123)
        z = tsne.fit_transform(x) 

        df = pd.DataFrame()
        df["y"] = true_diagnosis
        df["comp-1"] = z[:,0]
        df["comp-2"] = z[:,1]

        t = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                        palette=sns.color_palette("hls", 5),
                        data=df).set(title="Class Predictions") 
        figure = t.get_figure()    
        figure.savefig("./output/tsne.png", dpi=400)

        return