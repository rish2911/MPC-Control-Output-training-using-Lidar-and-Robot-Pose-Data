
from pipeline import MPCLearning
if __name__=="__main__":
    tr_path = "Input_data\\training"
    test_path = "Input_data\\testing"
    model = ["Reg", "NN", "SVM"]
    pipe = MPCLearning(tr_path, "SVM", test_path)