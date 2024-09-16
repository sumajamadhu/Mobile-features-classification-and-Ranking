from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open('mobile.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
    RM= float(request.values['RM'])
    CM= float(request.values['CM'])       
    BP= float(request.values['BP'])
    MT= float(request.values['MT'])
    SW= float(request.values['SW'])
    SH= float(request.values['SH'])
    IM= float(request.values['IM'])
    CS= float(request.values['CS'])
    TT= float(request.values['TT'])
    PW= float(request.values['PW'])
    PH= float(request.values['PH'])
    input=np.array([BP,CS,CM,IM,MT,PH,PW,RM,SH,SW,TT])
    input = np.reshape(input,(1, input.size))
    output=model.predict(input)
    print(output)
    for x in output:
        if (x==3):
            output='***Price range :: Very high cost *** Software and Hardware requirements :: Very high ***     Popularity Rank:: 3 - Medium ***'
        elif(x==2):
            output='***Price range :: High cost *** Software and Hardware requirements :: High ***    Popularity Rank ::  2- High ***'
        elif(x==1):
            output='***Price range :: Medium cost *** Software and Hardware requirements :: Medium ***    Popularity Rank:: 1- Very high ***'
        elif(x==0):
            output='***Price range :: Low cost *** Software and Hardware requirements ::  Low ***    Popularity Rank:: 3- Medium ***'    
    return render_template('result.html', prediction_text=' {}'.format(output)) # rendering the predicted result
if __name__=='__main__':
    app.run(port=5000)

