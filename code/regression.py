# %% [markdown]
# Importing libraries

# %%
import numpy as np
from numpy import array
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import sklearn.metrics  as sk
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler



# %%
class Simple_Regression:
    def __init__(self):
        self.weights_ = None
        pass


    def run_model(self, X_tr, y_tr, X_test, y_test):

        ###Preprocessing

        #training data
        self.X_train_og = X_tr
        self.y_train_og = y_tr

        #test data
        self.X_test = X_test
        self.y_test = y_test

        ###shuffle the data
        self.X_train_og, self.y_train_og = shuffle(self.X_train_og, self.y_train_og, random_state=0)
        self.X_test, self.y_test = shuffle(self.X_test, self.y_test, random_state=0)

        ###split data into training and validation
        X_train, X_val, y_train, y_val = train_test_split(self.X_train_og, self.y_train_og, test_size=0.2, random_state=42)
        print('output training size', y_train.shape)



        #PARAMETERS
        """try to put alpha for tuning"""
        k_all = range(0,10)
        k_list = []
        inscore_list_v = []
        valscore_list_v = []
        inscore_list_o = []
        valscore_list_o = []
        min_scorev = -100000
        min_scoreo = -100000
        final_kv = -1
        final_ko = 0

        for k in k_all:
          
            ###Scaling and training
            k/=1
            k_list.append(k)
            #model instance for velocity
            reg_v = make_pipeline(StandardScaler(), linear_model.Ridge(alpha=k))
            #model instance for omega
            reg_omega = make_pipeline(StandardScaler(), linear_model.Ridge(alpha=k))
            ###Velocity
            reg_v.fit(X_train, y_train[:,0])
            # w_b = reg_v.coef_
            ###Omega
            ###Velocity
            reg_omega.fit(X_train, y_train[:,1])
            # w_b_omega = reg_omega.coef_


            #IN SAMPLE TEST
            y_pred_train_v = reg_v.predict(X_train)
             # ###velocity range
            y_pred_train_v[y_pred_train_v>np.max(y_train[:,0])] = np.max(y_train[:,0])
            y_pred_train_v[y_pred_train_v<np.min(y_train[:,0])] = np.min(y_train[:,0])
            inscore_list_v.append(self.raw_score(y_train[:,0], y_pred_train_v))
            print('In sample score for velocity with k = ', k, ' \n', inscore_list_v[-1])

            ###omega range
            y_pred_train_o = reg_omega.predict(X_train)
            y_pred_train_o[y_pred_train_o>np.max(y_train[:,1])] = np.max(y_train[:,1])
            y_pred_train_o[y_pred_train_o<np.min(y_train[:,1])] = np.min(y_train[:,1])
            inscore_list_o.append(self.raw_score(y_train[:,1], y_pred_train_o))
            print('In sample score for omega with k = ', k, ' \n', inscore_list_o[-1])

            
            #VALIDATION
            # print(np.shape(A_mat_test))
            y_pred_val_v = reg_v.predict(X_val)
             # ###velocity range
            y_pred_val_v[y_pred_val_v>np.max(y_val[:,0])] = np.max(y_val[:,0])
            y_pred_val_v[y_pred_val_v<np.min(y_val[:,0])] = np.min(y_val[:,0])

            ###omega range
            y_pred_val_o = reg_omega.predict(X_val)
            y_pred_val_o[y_pred_val_o>np.max(y_val[:,1])] = np.max(y_val[:,1])
            y_pred_val_o[y_pred_val_o<np.min(y_val[:,1])] = np.min(y_val[:,1])


            #PERFORMANCE
            ###velocity
            valscore_list_v.append(self.raw_score(y_val[:,0], y_pred_val_v))
            #storing the hyperparameter value with least error
            if min_scorev<valscore_list_v[-1]:
                min_scorev=valscore_list_v[-1]
                final_kv = k
            print('Validation sample score for velocity with k = ', k, ' \n', valscore_list_v[-1])

            ###Omega
            valscore_list_o.append(self.raw_score(y_val[:,1], y_pred_val_o))
            if min_scoreo<valscore_list_o[-1]:
                min_scoreo=valscore_list_o[-1]
                final_ko = k
            print('Validation sample score for omega with k = ', k, ' \n', valscore_list_o[-1])


        
        #MERGING VALIDATION, RE-TRAINING AND TESTING
        ###Velocity
        reg_v = make_pipeline(StandardScaler(), linear_model.Ridge(alpha=final_kv))
        reg_v.fit(self.X_train_og, self.y_train_og[:,0])
        # w_bv = reg_v.coef_
        # print(w_bv)
        y_pred_test_v = reg_v.predict(self.X_test)
        y_pred_test_v[y_pred_test_v>np.max(self.y_test[:,0])] = np.max(self.y_test[:,0])
        y_pred_test_v[y_pred_test_v<np.min(self.y_test[:,0])] = np.min(self.y_test[:,0])
        outscore_v = (self.raw_score(self.y_test[:,0], y_pred_test_v))
        print('Out sample error for velocity with k = ', final_kv, ' \n', outscore_v)

        ###omega range
        reg_omega = make_pipeline(StandardScaler(), linear_model.Ridge(alpha=final_ko))
        reg_omega.fit(self.X_train_og, self.y_train_og[:,1])
        y_pred_test_o = reg_omega.predict(self.X_test)
        y_pred_test_o[y_pred_test_o>np.max(self.y_test[:,1])] = np.max(self.y_test[:,1])
        y_pred_test_o[y_pred_test_o<np.min(self.y_test[:,1])] = np.min(self.y_test[:,1])
        outscore_o = (self.raw_score(self.y_test[:,1], y_pred_test_o))
        print('Out sample error for omega with k = ', final_ko, ' \n', outscore_o)

        #PLOTTING
        #velocity
        self.hyperparameter_plot(k_list, valscore_list_v, inscore_list_v, 'validation', 'training')
        """Uncomment the part below for plotting learning curve"""
        # self.learning_curves(self.X_train_og, self.y_train_og[:,0], final_kv)
        ###Omega
        self.hyperparameter_plot(k_list, valscore_list_o, inscore_list_o, 'validation', 'training')
        """Uncomment the part below for plotting learning curve"""
        # self.learning_curves(self.X_train_og, self.y_train_og[:,1], final_ko)

    #conventional prediction function (not used!) 
    def prediction(self,W, A_mat):
        pred = A_mat.dot(W)
        return pred

    #function to plaot learning curve using cross validation
    def learning_curves(self, X,y, alpha):
        train_sizes, train_scores, test_scores = learning_curve(make_pipeline(StandardScaler(), linear_model.Ridge(alpha=alpha)), X, y, cv=10, \
            n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)

        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.subplots(1, figsize=(10,10))
        plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
        plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

        plt.title("Learning Curve")
        plt.xlabel("Training Set Size"), plt.ylabel("R2 Score"), plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    # to plot erros and score w.r.t to hyperparamters
    def hyperparameter_plot(self, k_list, score_list1, score_list2,label1='Validation', label2='Validation'):
        plt.plot(k_list, score_list1, label=label1)
        plt.plot(k_list, score_list2, label=label2) 
        plt.xlabel("Regularization parameter alpha or lambda")
        plt.ylabel("R2 Score")
        plt.legend()
        plt.show()


    # to find the scores and error in prediction
    def raw_score(self, y_true:np.array, y_pred:np.array)->float:
        #sum of square of residuals
        r2score = sk.r2_score(y_true, y_pred)
        msq = sk.mean_squared_error(y_true, y_pred)

        """ADD MORE"""

        return r2score