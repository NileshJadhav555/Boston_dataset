import numpy as np
import pandas as pd
import pickle
import json
import config


class BostonValue():
    def __init__(self,CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT):
        self.CRIM = CRIM
        self.ZN = ZN
        self.INDUS = INDUS
        self.CHAS = CHAS
        self.NOX = NOX
        self.RM = RM
        self.AGE = AGE
        self.DIS = DIS
        self.RAD = RAD
        self.TAX = TAX
        self.PTRATIO = PTRATIO
        self.B = B
        self.LSTAT = LSTAT

    def load_model(self):
        with open(config.MODEL_FILE_PATH,'rb') as f:
            self.model = pickle.load(f)

        with open(config.JSON_FILE_PATH,'r') as f:
            self.json_data = json.load(f)

    def Predicted_values(self):
        self.load_model()

        Test_array = np.zeros(len(self.json_data['columns']))
        Test_array[0] = self.CRIM
        Test_array[1] = self.ZN
        Test_array[2] = self.INDUS
        Test_array[3] = self.CHAS
        Test_array[4] = self.NOX
        Test_array[5] = self.RM
        Test_array[6] = self.AGE
        Test_array[7] = self.DIS
        Test_array[8] = self.RAD
        Test_array[9] = self.TAX
        Test_array[10] = self.PTRATIO
        Test_array[11] = self.B
        Test_array[12] = self.LSTAT


        print("Test array:",Test_array)  # 9 values
        predicted_values  = self.model.predict([Test_array])
        return predicted_values
