from flask import Flask, render_template,request,jsonify
import config
from Project_app.utils import BostonValue


####################################### HOME API #####################################################

app = Flask(__name__)

@app.route('/')
def hello_flask():
    print("WELCOME to Flask")
    return jsonify({'Model':"Sucessfully builded"})

#######################################################################################################

@app.route("/predict_values")
def get_predicted_values():

    CRIM = 0.008
    ZN = 25
    INDUS = 2.5
    CHAS = 0
    NOX = 0.8
    RM = 6
    AGE = 47
    DIS = 5.8
    RAD = 2
    TAX = 150
    PTRATIO = 17.5
    B = 456
    LSTAT = 3.78

    boston = BostonValue(CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT)

    value = boston.Predicted_values()

    return jsonify({'Result':f"Predicted value is :{value}"})


if __name__ == '__main__':
    app.run()
