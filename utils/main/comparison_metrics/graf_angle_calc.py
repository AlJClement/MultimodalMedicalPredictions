import numpy as np
import math as math
from matplotlib import pyplot as plt
import os
import math
class graf_angle_calc():
    def __init__(self) -> None:
        '''
        landmarks list should contain 5 points x,y. 
        with order: illium 1, illium 2, bony rim, lower limb point, labrum
        grf_dict: where a is alpha angle, and d is discription'''
        self.grf_dic = {
            "i": {'a':'>=60', 'd': 'Normal: Discharge Patient'},
            "ii": {'a':'43-60', 'd': 'Rescan +/- brace'},
            "iii/iv": {'a':'<43', 'd':'Abnormal: Clinical Review + treat'},
            "Nan": {'a':'<43', 'd':'Alpha Not Predicted, landmark on same point'},
            }

        pass

    def get_landmarks(self, landmarks: list, flip_axis = True):
        #landmarks should be list
        #ilium
        i1 = [float(i) for i in landmarks[0]]
        i2 = [float(i) for i in landmarks[1]]
        #bonyrim
        br = [float(i) for i in landmarks[2]]
        #lower limb point
        ll = [float(i) for i in landmarks[3]]
        #labrum
        l = [float(i) for i in landmarks[4]]
        return i1,i2,br,ll,l

    def get_alpha_category(self, alpha:float):
        #print('alpha is', alpha)
        if alpha >= 60:
            return '>=60'
        elif alpha > 43 and alpha < 60:
            return'43-60'
        elif alpha < 43:
            return'<43'
        elif np.isnan(alpha):
            return 'Nan'
        else:
            raise ValueError

    def get_alpha_class(self, alpha: float):
        '''get classification and discription from dictionary based on, angle'''
        alpha = self.get_alpha_category(alpha)
        print(alpha)
        for key in self.grf_dic.items(): 
            if self.grf_dic[key[0]]['a'] == alpha:
                graf_class = key[0]
                graf_discription = self.grf_dic[key[0]]['d']
            
        return graf_class, graf_discription

    def plot_theta(self,intersection, a, start_vector, direction='clockwise'):
        r = 50
        x0,y0 = intersection

        theta1 = (np.arctan(start_vector[0]/start_vector[1]))
        theta2 = theta1+math.radians(a)

        if direction == 'clockwise':
            t = np.linspace(theta1, theta2, 11)
        else:
            t = np.linspace(theta2+180, theta1+180, 11)

        x = r*np.cos(t) + x0
        y = r*np.sin(t) + y0
        xt = x[6]+5
        yt = y[6]+5

        return x, y, xt, yt
    
    def get_vector(self,p1,p2):
        x1,y1 = p1
        x2,y2 = p2
        v=[x2-x1,y2-y1]
        return v
    
    def get_intersection(self, vec1, vec2):
        #find slope and intersect for each vector
        m1 = (vec1[0][0]-vec1[0][1])/(vec1[1][0]-vec1[1][1])
        b1 = vec1[0][0]-m1*(vec1[0][1])
        m2 = (vec2[0][0]-vec2[0][1])/(vec2[1][0]-vec2[1][1])
        b2 = vec2[0][0]-m2*(vec2[0][1])

        xi = (b1 - b2) / (m2 - m1)
        yi = m1 * xi + b1

        intersection = [xi, yi]
        return intersection
    
    def plot_landmarks(self, landmarks):
        i1, i2, br, ll, l = landmarks
        #plot landmarks
        plt.scatter(i1[0], i1[1],c='r', s=20)
        plt.scatter(i2[0], i2[1], c='r', s=20)
        plt.scatter(l[0], l[1], c='b', s=20)
        plt.scatter(ll[0], ll[1], c='g', s=20)
        plt.scatter(br[0], br[1], c='y', s=20)
        return
    
    def calculate_alpha(self, landmarks: list, plot = False):
        ''' This function takes a list of 5 landmarks and calculates the alpha angle based on the labrum vector and bony rim
        landmarks list should contain 5 points x,y. 
        with order: illium 1, illium 2, bony rim, lower limb point, labrum
        v_baseline: is the vector along the labrum
        v_bonyroof: vector along bony rim
        beta angle calculations commented out'''
        if len(landmarks) != 5:
            return ValueError('There are not only 5 landmarks')
        
        i1, i2, br, ll, l = self.get_landmarks(landmarks, flip_axis = True)
        #print(i1, i2, br, ll, l)
        
        v_baseline = self.get_vector(i1,i2)
        #print(v_baseline)
        #v_cartroof = self.get_vector(self.br,self.l)
        v_bonyroof = self.get_vector(br,ll)
        #print(v_bonyroof)

        #angles using arccosbeta
        try:
            a_rad = np.arccos(np.dot(v_baseline,v_bonyroof)/(np.linalg.norm(v_baseline)*np.linalg.norm(v_bonyroof)))
            alpha = math.degrees(a_rad)
            alpha = a_rad*180/np.pi
            if alpha > 90:
                alpha = alpha - 90
        except:
            a_rad = 0.0
            alpha = 0.0
            print(np.dot(v_baseline,v_bonyroof),(np.linalg.norm(v_baseline),np.linalg.norm(v_bonyroof)))
            print('arc cos did not work')


        #b_rad = np.arccos(np.dot(v_baseline,v_cartroof)/(np.linalg.norm(v_baseline)*np.linalg.norm(v_cartroof)))
        #b = math.degrees(b_rad)

        if plot==True:
            self.plot_landmarks(landmarks)
            ##PLOTTING ARC##
            intersection = self.get_intersection([i1,i2],[br,ll])
            x, y, xt, yt = self.plot_theta(intersection, alpha, v_baseline)
            plt.plot(x,y,color='w',linewidth=0.5)
            plt.text(xt,yt,'a='+str(round(alpha))+u"\u00b0",color='w')
            plt.show()

        return alpha
    
    def graf_class_comparison(self, pred, pred_map, true, true_map, pixelsize):
        'pred and true are tensors, so convert to numpy'
        pred=pred.detach().cpu().numpy()
        true=true.detach().cpu().numpy()

        alpha_pred = self.calculate_alpha(pred)
        class_pred = self.get_alpha_class(alpha_pred)
        print('pred:', alpha_pred)

        alpha_true = self.calculate_alpha(true)
        class_true = self.get_alpha_class(alpha_true)
        print('true:',alpha_true)
        
        alpha_diff = alpha_pred-alpha_true

        ls_values = [['alpha pred', alpha_pred],
                    ['class pred', class_pred[0]],
                    ['alpha true', alpha_true],
                    ['class true', class_true[0]],
                    ['difference alpha', alpha_diff]
                    ]
     
        return ls_values