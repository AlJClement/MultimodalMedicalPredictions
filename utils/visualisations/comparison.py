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

    def confusion_matrix_multiclass(self, classes, confusion_matrix_multiclasses,loc='test',name =''):
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
            ax[c].set_title('Confusion Matrix: '+ name)
        
        plt.savefig('./output/'+loc+'/Confusion_Matrix_allclasses'+str(c)+'.png')

    def to_one_hot(self, y, num_classes):
        y = y.squeeze().astype(int)
        hot = np.eye(num_classes)[y]
        return hot

    def true_vs_pred_scatter(self, pred, true, loc='test'):
        #plot predicted vs true value outputs.
        #order by true
        plt.clf()
        dataset = pd.DataFrame({'pred': pred, 'true': true}, columns=['pred', 'true'])

        #plot theta vs theta true vs pred
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_aspect('equal', adjustable='box')

        plt.scatter(dataset['true'], dataset['pred'], c='b', alpha=1)
        plt.axis('equal')
        x, y =dataset['true'], dataset['pred']
        #add line of best fit
        coef = np.polyfit(x,y,1)
        poly1d_fn = np.poly1d(coef) 
        m, b = np.polyfit(x, y, 1)
        plt.plot(x,y, 'yo', x, poly1d_fn(x), '--k') #'--k'=black dashed line, 'yo' = yellow circle marker
        plt.xlabel('True Graf Angle')
        plt.ylabel('Predicted Graf Angle')
        #ax.set_aspect('equal', adjustable='box')
        plt.savefig('./output/'+loc+'/true_vs_pred.png')
        #sort to plot so we can see thresholds        #plot true and pred

        plt.clf()
        fig2 = plt.figure()
        ax2 = fig2.add_subplot()
        dataset = dataset.sort_values('true')
        dataset.reset_index(drop=True)
        patient = range(len(dataset))
        plt.scatter(patient,dataset['true'], c='g', alpha=1)
        plt.scatter(patient,dataset['pred'], c='r', alpha=0.5)
        plt.xlabel('Patient')
        plt.ylabel('Angle')

        for thresh in self.threshold_list:
            plt.axhline(y=thresh, color='b', linestyle='--')

        plt.savefig('./output/'+loc+'/true_vs_pred_bypatient.png')

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