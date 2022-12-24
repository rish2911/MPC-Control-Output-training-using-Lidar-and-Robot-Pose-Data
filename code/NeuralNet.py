import numpy as np
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import sklearn.metrics  as sk
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

class NeuralNet():
    def __init__(self) -> None:
        self.model_ = None
        self.weights_ = None
        pass

    def run_model(self, X_tr, y_tr, X_test, y_test):
        ###Preprocessing of data

        #training data
        self.X_train_og = X_tr
        self.y_train_og = y_tr
    
        #test data
        self.X_test = X_test
        self.y_test = y_test

        ###shuffle the data
        self.X_train_og, self.y_train_og = shuffle(self.X_train_og, self.y_train_og, random_state=0)
        self.X_test, self.y_test = shuffle(self.X_test, self.y_test, random_state=0)

        print('testing size', self.X_train_og.shape, self.y_train_og)
        
        ###split data into training and validation
        X_train, X_val, y_train, y_val = train_test_split(self.X_train_og, self.y_train_og, test_size=0.2, random_state=42)
        print('output training size', y_train.shape)

        #PARAMETERS
        k_all = range(1,10)
        k_list = []
        inscore_list_v = []
        valscore_list_v = []
        inscore_list_o = []
        valscore_list_o = []
        min_scorev = 10000
        min_scoreo = 10000
        final_kv = 0
        final_ko = 0
        for k in k_all:

            k/=1

            ##Scaling and training
            k_list.append(k)
            #model instance for velocity
            reg_v = make_pipeline(StandardScaler(), MLPRegressor(solver='lbfgs', alpha=k, hidden_layer_sizes=(10, 2), random_state=1, max_iter=500))
            #model instance for omega
            reg_omega = make_pipeline(StandardScaler(), MLPRegressor(solver='lbfgs', alpha=k, hidden_layer_sizes=(10, 2), random_state=1, max_iter=500))
            reg_v.fit(X_train, y_train[:,0])
            reg_omega.fit(X_train, y_train[:,1])

            #IN SAMPLE TEST
            y_pred_train_v = reg_v.predict(X_train)
            ###velocity range
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
            y_pred_val_v = reg_v.predict(X_val)
            ###velocity range
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
            print('Validation sample score for omega with k = ', k/100, ' \n', valscore_list_o[-1])
        

        #MERGING VALIDATION, RE-TRAINING AND TESTING
        ##Velocity
        reg_v = make_pipeline(StandardScaler(), MLPRegressor(solver='lbfgs', alpha=final_kv, hidden_layer_sizes=(10, 2), random_state=1, max_iter=500))
        reg_v.fit(self.X_train_og, self.y_train_og[:,0])
        print("scalarscoring function for v", reg_v.score(self.X_test, self.y_test[:,0]))
        y_pred_test_v = reg_v.predict(self.X_test)
        y_pred_test_v[y_pred_test_v>np.max(self.y_test[:,0])] = np.max(self.y_test[:,0])
        y_pred_test_v[y_pred_test_v<np.min(self.y_test[:,0])] = np.min(self.y_test[:,0])
        outscore_v = (self.raw_score(self.y_test[:,0], y_pred_test_v))
        print('Out sample error for velocity with k = ', final_kv, ' \n', outscore_v)
        
        ###omega range
        reg_omega = make_pipeline(StandardScaler(), MLPRegressor(solver='lbfgs', alpha=final_ko, hidden_layer_sizes=(10, 2), random_state=1, max_iter=500))
        reg_omega.fit(self.X_train_og, self.y_train_og[:,1])
        print("scalarscoring function for w",reg_v.score(self.X_test, self.y_test[:,1]))
        y_pred_test_o = reg_omega.predict(self.X_test)
        y_pred_test_o[y_pred_test_o>np.max(self.y_test[:,1])] = np.max(self.y_test[:,1])
        y_pred_test_o[y_pred_test_o<np.min(self.y_test[:,1])] = np.min(self.y_test[:,1])
        outscore_o = (self.raw_score(self.y_test[:,1], y_pred_test_o))
        print('Out sample error for omega with k = ', final_ko, ' \n', outscore_o)
        
        #PLOTTING
        self.hyperparameter_plot(k_list, valscore_list_v, inscore_list_v, 'validation v','training v')
        self.hyperparameter_plot(k_list, valscore_list_o, inscore_list_o, 'validation w','training w')

        """Uncomment the part below for plotting learning curve"""
        # self.learning_curves(self.X_train_og[:100000], self.y_train_og[:100000,0], self.model_, final_kv)
        # self.learning_curves(self.X_train_og[:100000], self.y_train_og[:100000,1], self.model_, final_ko)
        pass
    
    #conventional prediction function (not used!)
    def prediction(self,W, A_mat):
        pred = A_mat.dot(W)
        return pred

    #function to plaot learning curve using cross validation
    def learning_curves(self, X,y, kernel:str, final_c:float):
        train_sizes, train_scores, test_scores = learning_curve(make_pipeline(StandardScaler(), MLPRegressor(solver='lbfgs', alpha=final_c,\
            hidden_layer_sizes=(5, 2), random_state=1)), X, y, cv=10, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))
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
    def hyperparameter_plot(self, k_list, score_list1, score_list2,label1='', label2=''):
        plt.plot(k_list, score_list1, label=label1)
        plt.plot(k_list, score_list2, label=label2) 
        plt.xlabel("Regularization parameter alpha or lambda")
        plt.ylabel("MSE")
        plt.legend()
        plt.show()

    # to find the scores and error in prediction
    def raw_score(self, y_true:np.array, y_pred:np.array)->float:
        #sum of square of residuals
        r2score = sk.r2_score(y_true, y_pred)
        msq = sk.mean_squared_error(y_true, y_pred)


        """ADD MORE IF NEEDED"""

        return msq