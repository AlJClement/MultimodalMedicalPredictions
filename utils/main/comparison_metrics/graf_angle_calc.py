import numpy as np
import math as math
from matplotlib import pyplot as plt
import os
import math
class graf_angle_calculations():
    def __init__(self) -> None:
        '''
        landmarks list should contain 5 points x,y. 
        with order: illium 1, illium 2, bony rim, lower limb point, labrum
        grf_dict: where a is alpha angle, and d is discription'''
        self.grf_dic = {
            "i": {'a':'>=60', 'd': 'Normal: Discharge Patient'},
            "ii": {'a':'43-60', 'd': 'Rescan +/- brace'},
            "iii/iv": {'a':'<43', 'd':'Abnormal: Clinical Review + treat'},
            }

        pass

    def get_landmarks(self, landmarks):
        #ilium
        i1 = [float(i) for i in landmarks[0].strip('\n').split(',')]
        i2 = [float(i) for i in landmarks[1].strip('\n').split(',')]
        #bonyrim
        br = [float(i) for i in landmarks[2].strip('\n').split(',')]
        #lower limb point
        ll = [float(i) for i in landmarks[3].strip('\n').split(',')]
        #labrum
        l = [float(i) for i in landmarks[4].strip('\n').split(',')]
        return i1,i2,br,ll,l

    def get_alpha_category(self, alpha:float):
        if alpha >= 60:
            return '>=60'
        elif alpha > 43 and alpha < 60:
            return'43-60'
        elif alpha < 43:
            return'<43'
        else:
            raise ValueError

    def get_alpha_class(self, alpha: float):
        '''get classification and discription from dictionary based on, angle'''
        alpha = self.get_alpha_category(alpha)
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
        i1, i2, br, ll, l = self.get_landmarks_from_ls(landmarks)
        #plot landmarks
        plt.scatter(i1[1], i1[0],c='r', s=20)
        plt.scatter(i2[1], i2[0], c='r', s=20)
        plt.scatter(l[1], l[0], c='b', s=20)
        plt.scatter(ll[1], ll[0], c='g', s=20)
        plt.scatter(br[1], br[0], c='y', s=20)
        return
    
    def calculate_alpha(self, landmarks: list, plot = True):
        ''' This function takes a list of 5 landmarks and calculates the alpha angle based on the labrum vector and bony rim
        landmarks list should contain 5 points x,y. 
        with order: illium 1, illium 2, bony rim, lower limb point, labrum
        v_baseline: is the vector along the labrum
        v_bonyroof: vector along bony rim
        beta angle calculations commented out'''
        if len(landmarks) != 5:
            return ValueError('There are not only 5 landmarks')
        i1, i2, br, ll, l = self.get_landmarks_from_ls(landmarks)

        v_baseline = self.get_vector(i1,i2)
        #v_cartroof = self.get_vector(self.br,self.l)
        v_bonyroof = self.get_vector(br,ll)
        #angles using arccosbeta
        a_rad = np.arccos(np.dot(v_baseline,v_bonyroof)/(np.linalg.norm(v_baseline)*np.linalg.norm(v_bonyroof)))
        alpha = math.degrees(a_rad)

        #b_rad = np.arccos(np.dot(v_baseline,v_cartroof)/(np.linalg.norm(v_baseline)*np.linalg.norm(v_cartroof)))
        #b = math.degrees(b_rad)

        if plot==True:
            self.plot_landmarks()
            ##PLOTTING ARC##
            intersection = self.get_intersection(v_baseline,v_bonyroof)
            x, y, xt, yt = self.plot_theta(intersection, alpha, v_baseline)
            plt.plot(x,y,color='w',linewidth=0.5)
            plt.text(xt,yt,'a='+str(round(alpha))+u"\u00b0",color='w')
            plt.show()

        return alpha
    
    def graf_class_comparison(self, pred, true):

        alpha_pred = self.calculate_alpha(pred)
        class_pred = self.get_alpha_class(alpha_pred)

        alpha_true = self.calculate_alpha(true)
        class_true = self.get_alpha_class(alpha_true)
        
        return [alpha_pred, class_pred, alpha_pred, class_true]