import numpy as np
import perceptron as perc
import regression as reg
import NeuralNet as nn
import SVM as supp
from utils import euler_from_quaternion
from data_import import data_imp
import sklearn.metrics  as sk
import xg_boost as xg



'''Note Lidar data is for 270 deg scan'''
class MPCLearning():
    def __init__(self, tr_data_path:str=None, model:str=None, test_data_path:str=None):
        # self.test_data_ = test_data
        self.data_ = None
        self.tr_data_path_ = tr_data_path
        self.test_data_path_ = test_data_path


        """call the pipeline below this, basically training 
        and testing functions"""
        if tr_data_path != None:
            A_train, y_train = self.run_training_model(self.tr_data_path_)

        if test_data_path != None:
            A_test, y_test = self.run_testing_model(self.test_data_path_)

        if model != None:
            model_map = {"Reg": reg.Simple_Regression(),\
                "NN":nn, "SVM":supp.SupportVectorMachine(), "XG":xg.XGBooster()}
            self.model_ = model_map[model]

            self.model_.run_model(A_train, y_train, A_test, y_test)
        

    def data_input_cleaning(self, data_path)->np.array:
        """put all the cleaning algorithms here"""
        
        imported = data_imp(data_path)
        row, col = imported.data_.shape
        self.data_ = np.zeros((row, col))
        row_list = list()
        # print('hi')
        for i in range(imported.data_.shape[0]):
            try:
                self.data_[i]=  np.array(imported.data_[i,:], dtype=np.float64)
            except:
                row_list.append(i)
                continue
        return np.delete(self.data_,row_list[:], axis=0)
        pass

    def extract_columndata(self, input_data):
        """can be called by both training and testing functions"""
        lidar_data = input_data[:,:1080]
        final_xy = input_data[:,1080:1082]
        final_quat = input_data[:,1082:1084]
        local_xy = input_data[:,1084:1086]
        local_quat = input_data[:,1086:1088]
        bot_xy = input_data[:,1088:1090]
        bot_quat = input_data[:,1090:1092]
        output = input_data[:,1092:1094]

        return lidar_data, final_xy, final_quat, local_xy, local_quat,\
            bot_xy, bot_quat, output

    def quat_to_euler(self, local_quat, final_quat, bot_quat):
        
        """return the values for yaw angle w.r.t to x - axis"""
        """can be called by both training and testing functions"""
        """first two values are zero because only yaw is considered"""
        _,_,local_yaw = euler_from_quaternion(0, 0, local_quat[:,0], local_quat[:,1])
        _,_,final_yaw = euler_from_quaternion(0, 0, final_quat[:,0], final_quat[:,1])
        _,_,bot_yaw = euler_from_quaternion(0, 0, bot_quat[:,0], bot_quat[:,1])

        ####conversion into 0 to 360 deg
        local_yaw = np.rad2deg(local_yaw)
        local_yaw[local_yaw<0] = 360 + local_yaw[local_yaw<0]
        final_yaw = np.rad2deg(final_yaw)
        final_yaw[final_yaw<0] = 360 + final_yaw[final_yaw<0]
        bot_yaw = np.rad2deg(bot_yaw)
        bot_yaw[bot_yaw<0] = 360 + bot_yaw[bot_yaw<0]
        return local_yaw, final_yaw, bot_yaw


    

    ####function for features
    def distance_from_goal(self, x_bot:np.array, y_bot:np.array, \
        x_local:np.array, y_local:np.array, x_final:np.array\
            , y_final:np.array)-> np.array:
        return np.sqrt((x_bot-x_local)**2 + (y_bot-y_local)**2), \
                np.sqrt((x_bot-x_final)**2 + (y_bot-y_final)**2)


    ###Note: Verify that angle units are same and avoid occurence of infinity
    def angle_diff_goal(self, bot_pose:np.array, x_bot:np.array, y_bot:np.array, \
        x_local:np.array, y_local:np.array, x_final:np.array, y_final:np.array)->None:

        ####check the closest angle
        rows = x_local.shape[0]
        target_local_ang = np.zeros((rows,1))
        target_final_ang = np.zeros((rows,1))
        for i in range(x_local.shape[0]):
            # print(x_local[i])   
            x_diff = x_local[i] - x_bot[i]
            y_diff = y_local[i] - y_bot[i]
            if x_diff!=0:
                target_local_ang[i] = np.rad2deg(np.arctan(y_diff/x_diff))
                target_final_ang[i] = np.rad2deg(np.arctan(y_diff/x_diff))
            else:
                target_local_ang[i] = np.rad2deg(np.arctan(np.sign(y_diff)*np.inf))
                target_final_ang[i] = np.rad2deg(np.arctan(np.sign(y_diff)*np.inf))
        target_local_ang[target_local_ang<0] = 360+target_local_ang[target_local_ang<0]
        target_final_ang[target_final_ang<0] = 360+target_final_ang[target_final_ang<0]
        
        self.local_angdiff = bot_pose.reshape((rows,1))-target_local_ang
        self.final_angdiff = bot_pose.reshape((rows,1)) - target_final_ang
        # print((self.final_angdiff))


         ####return the closest angle c/w or ac/w 
        return None

    ####Lidar data usage

    def distance_in_front(self, lidar_data:np.ndarray)->np.array:
        return lidar_data[:,540]

    def space_towards_traj(self, lidar_data:np.array)->np.array:

        ####since 0.25 degree is the resolution
       self.local_los = 4*np.array((np.rad2deg(self.local_angdiff)), dtype=int)
       self.final_los = 4*np.array((np.rad2deg(self.final_angdiff)), dtype=int)
       rows = self.local_los.shape[0]
       localtraj_lid = np.zeros((rows,1))
       finaltraj_lid = np.zeros((rows,1))
       for i in range(rows):
        if self.local_los[i]>=540 or self.local_los[i]<=-540:
            localtraj_lid[i] = 0
            finaltraj_lid[i] = 0
        else:
            localtraj_lid[i] = lidar_data[i,540+self.local_los[i]]
            finaltraj_lid[i] = lidar_data[i,540+self.final_los[i]]
            
       return localtraj_lid, finaltraj_lid

    def yaw_diff(self, bot_yaw, local_yaw, final_yaw):
        """its basically the difference in yaw angles """
        return np.abs(bot_yaw-local_yaw), np.abs(bot_yaw-final_yaw)
        pass


    def minimum_corridor(self, lidar_data:np.array)->np.array:
        '''return the minimum distance the robot can move towards local goal
        or final goal along the LOS considering some width of the bot'''
        width_robot  = 30 #cm
        lidar_value_l, lidar_value_f = self.space_towards_traj(lidar_data)
        ###calculation for angle of interest w.r.t to trajectory to be followed
        
        ###maximum distance possible towards local goal trajectory
        self.local_los[self.local_los>=540] = 540
        self.local_los[self.local_los<=540] =-540
        self.final_los[self.local_los>=540] = 540
        self.final_los[self.local_los<=540] =-540
        rows = self.local_los.shape[0]
        max_local = np.zeros((rows,1))
        max_final = np.zeros((rows,1))
        # alpha_local = np.zeros((rows,1))
        # alpha_final = np.zeros((rows,1))
        for i in range(rows):
            # print(lidar_value_l[i])
            if lidar_value_l[i]==0:
                alpha_local = int(np.abs(np.rad2deg(np.arctan(np.inf))))
                
                
            else:

                alpha_local = int(np.abs(np.rad2deg(np.arctan(0.6*width_robot/lidar_value_l[i]))))
            if lidar_value_f[i]==0:    
                alpha_final = int(np.abs(np.rad2deg(np.arctan(np.inf))))
            else:
                alpha_final = int(np.abs(np.rad2deg(np.arctan(0.6*width_robot/lidar_value_f[i]))))
            if alpha_local<0:
                    alpha_local = 360+alpha_local
            if alpha_final<0:
                    alpha_final = 360+alpha_final
            range_low_loc = int(540+self.local_los[i]-4*alpha_local)
            range_up_loc = int(540+self.local_los[i]+4*alpha_local)
            if range_low_loc<0:
                range_low_loc= 0
            if range_up_loc>=1080:
                range_up_loc=1080
            range_low_fin = int(540+self.final_los[i]-4*alpha_final)
            range_up_fin = int(540+self.final_los[i]+4*alpha_final)
            if range_low_fin<0:
                range_low_fin= 0
            if range_up_fin>=1080:
                range_up_fin=1080 
            max_local[i] = np.min(lidar_data[i,range_low_loc:range_up_loc])
            ###maximum distance possible towards final goal trajectory
            max_final[i] = np.min(lidar_data[i,range_low_fin:range_up_fin])
        return max_local, max_final


    def get_features(self, lidar_data, final_xy, final_ang, \
        local_xy, local_ang, bot_xy, \
        bot_ang):
        f_list = []
        ###feature1 and feature 2
        x1,x2 = self.distance_from_goal(bot_xy[:,0], bot_xy[:,1], local_xy[:,0], local_xy[:,1], \
            final_xy[:,0], final_xy[:,1])
        f_list.append(x1/np.mean(x1))
        f_list.append(x2/np.mean(x2))
  

        ###feature3 and feature 4    
        self.angle_diff_goal(bot_ang, bot_xy[:,0], bot_xy[:,1], local_xy[:,0], local_xy[:,1], \
            final_xy[:,0], final_xy[:,1])
        x3 = self.local_angdiff
        x4 =self.final_angdiff
        f_list.append(x3/np.mean(x3))
        f_list.append(x4/np.mean(x4))


        ###feature5
        x5 = self.distance_in_front(lidar_data)
        f_list.append(x5/np.mean(x5))


        ###feature 6 and 7
        x6, x7 = self.space_towards_traj(lidar_data)
        # print(x6)
        f_list.append(x6/np.mean(x6))
        f_list.append(x7/np.mean(x7))

        ###feature 8 and 9
        x8, x9 = self.yaw_diff(bot_ang, local_ang, final_ang)
        f_list.append(x8/np.mean(x8))
        f_list.append(x9/np.mean(x9))


        # ###feature 10 and 11
        x10, x11 = self.minimum_corridor(lidar_data)
        f_list.append(x10/np.mean(x10))
        f_list.append(x11/np.mean(x11))



        ###reshaping and size check
        for j,i in enumerate(f_list):
            i = np.reshape(i,(np.shape(i)[0],1))
            # print('shape of feature x'+str(j+1),np.shape(i))
        # f_list = np.stack(f_list, axis=0 )
        # print(np.shape(x1)[0])
        """the expression is ugly, find a better way"""
        A_mat = np.hstack((np.ones((np.shape(x1)[0],1)).reshape(np.shape(x1)[0],1),x1.reshape(np.shape(x1)[0],1),\
             x2.reshape(np.shape(x1)[0],1),x3.reshape(np.shape(x1)[0],1),x4.reshape(np.shape(x1)[0],1),\
                x5.reshape(np.shape(x1)[0],1),x6.reshape(np.shape(x1)[0],1),x7.reshape(np.shape(x1)[0],1),\
                    x8.reshape(np.shape(x1)[0],1),x9.reshape(np.shape(x1)[0],1),x10.reshape(np.shape(x1)[0],1),\
                        x11.reshape(np.shape(x1)[0],1)))
        
        print(A_mat.shape)
        

        return A_mat

    def some_cost_function(self):
        """some MPC cost function like max velocity, max w, max steering angle"""
        pass

    #### models and regularization, just provide the path of data
    def run_training_model(self, data_path:str)->None:
        """desired algorithm will be called and return the trained weights and curve"""

        ###what will be data???? which curve??

        tr_data_ = self.data_input_cleaning(data_path)
        """do data splitting here itself then do the rest of the things"""

        lidar_data, final_xy, final_quat, \
        local_xy, local_quat, bot_xy, \
        bot_quat, output = self.extract_columndata(tr_data_)

        ###conversion into yaw angles using quaternions
        local_yaw, final_yaw, bot_yaw =self.quat_to_euler(local_quat, final_quat, bot_quat)

        ###collecting features
        A_mat = self.get_features(lidar_data, final_xy, final_yaw, \
        local_xy, local_yaw, bot_xy, \
        bot_yaw)
        return A_mat, output
        # # print(np.shape(output))
        # ##weights for v
        # self.model_.training(A_mat, output[:,0])
        # self.weights_v = self.model_.weights_

        # ##weights for w
        # self.model_.training(A_mat, output[:,1])
        # self.weights_w = self.model_.weights_
        # print("size of weights ", np.shape(self.weights_w), '\n')

        # pred_output_v = A_mat.dot((self.weights_v.reshape(np.shape(self.weights_v)[0],1)))
        # print("size of pred v ", np.shape(pred_output_v), '\n')
        # pred_output_w = A_mat.dot((self.weights_w.reshape(np.shape(self.weights_w)[0],1)))
        # print("size of pred w ", np.shape(pred_output_w), '\n')




        # ###velocity range
        # pred_output_v[pred_output_v>np.max(output[:,0])] = np.max(output[:,0])
        # pred_output_v[pred_output_v<np.min(output[:,0])] = np.min(output[:,0])

        # ###omega range
        # pred_output_w[pred_output_w>np.max(output[:,1])] = np.max(output[:,1])
        # pred_output_w[pred_output_w<np.min(output[:,1])] = np.min(output[:,1])
        
        # score_v = self.raw_score(output[:,0], pred_output_v)
        # print("R2 insample score for v prediction is ", score_v, '\n')

        # score_w = self.raw_score(output[:,1], pred_output_w)
        # print("R2 insample score for w prediction is ", score_w, '\n')


        # '''finally needed is'''
        # # self.weights, self.insamplerr, self.in_misc_data =  self.model_.training(tr_input, tr_output)
        # pass

    def run_testing_model(self, data_path:str):
        """desired algorithm will be called and return the epoch"""
        test_data_ = self.data_input_cleaning(data_path)
        lidar_data, final_xy, final_quat, \
        local_xy, local_quat, bot_xy, \
        bot_quat, output = self.extract_columndata(test_data_)

        ###conversion into yaw angles using quaternions
        local_yaw, final_yaw, bot_yaw =self.quat_to_euler(local_quat, final_quat, bot_quat)
        ###collecting features
        A_mat = self.get_features(lidar_data, final_xy, final_yaw, \
        local_xy, local_yaw, bot_xy, \
        bot_yaw)
        return A_mat, output




    def raw_score(self, y_true:np.array, y_pred:np.array)->float:
        #sum of square of residuals
        r2score = sk.r2_score(y_true, y_pred)
        msq_error = sk.mean_squared_error(y_true, y_pred)
        # max_error = sk.max_error(y_true, y_pred)

        return [r2score, msq_error] 
