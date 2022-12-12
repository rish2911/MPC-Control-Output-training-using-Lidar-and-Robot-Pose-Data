from sklearn.linear_model import Perceptron

class Perceptron_model():
    def __init__(self):
        self.percept_ = Perceptron()

    def training(self,x_data, y_data):
        self.percept_.fit(x_data, y_data)
        self.weights_ = self.percept_.coef_
        self.error = self.percept_.loss_function_

    # def testing(self, x_data):
    #     x_data*self.weights
