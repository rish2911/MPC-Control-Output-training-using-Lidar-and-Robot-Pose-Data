from pipeline import MPCLearning #importing the main implementation
if __name__=="__main__":
    tr_path = "Input_data\\training"  # path to training input folder
    test_path = "Input_data\\testing" # path to testing input folder
    model = ["Reg", "NN", "SVM", "XG"] # implemented models

    try:    
        while True:
        
            user_input = input("What model do you wwant to use, type  \
            \n XG for XGBoost, \n NN for Neural Network (MLP), \n Reg for linear regression, \n SVM for Support Vectors \n")
            if user_input in model:
                pipe = MPCLearning(tr_path, user_input, test_path)
                break
            else:
                print("wrong input, try again \n")
    except:
            print("Create input folder and put files as specified in readme \n")
