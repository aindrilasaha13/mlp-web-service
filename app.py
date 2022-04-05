from flask import Flask,send_file
import pandas as pd,numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

app = Flask(__name__)

@app.get("/")
def lr_predict():
    
    df = pd.read_excel(r"datasets/covid_daily_data_of_India.xlsx",parse_dates=True)
    df1 = df.copy()
    
    cols = ["date","new_cases"]
    for col in df1.columns:
        if col not in cols:
            df1.drop(col, axis=1, inplace=True)

    df1 = df1.tail(60)
    df1.set_index('date', inplace=True, drop=True)
    df1["new_cases"]=df1["new_cases"].replace(0,df1["new_cases"].mean())

    for i in range(1,8):
        col = []
        for j in range(i):
            col.append(0)
        for val in df1["new_cases"]:
            col.append(val)
        prev_new_cases = col[0:len(col)-i]
        df1.insert(0,"(t-"+str(i)+")th day",prev_new_cases,True)
    
    X = df1.drop("new_cases",axis = 1, inplace = False)
    Y = df1["new_cases"]
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,shuffle=False)

    mlp_model = MLPRegressor(hidden_layer_sizes=(64,64,64),activation="relu" ,random_state=1, max_iter=2000)
    mlp_model.fit(X_train,Y_train)
    forecast = mlp_model.predict(X_test)
    actual = Y_test
    df_test_res = pd.DataFrame( np.c_[forecast,actual], index = X_test.index, columns = ["forecast","actual"] )
    df_test_res["forecast"].apply(np.ceil)
    df_test_res["actual"].apply(np.ceil)
    
    x = df_test_res.index
    y_actual = df_test_res["actual"]
    y_forecast = df_test_res["forecast"]
    calc_mape = np.mean(np.abs(y_forecast - y_actual)/np.abs(y_actual))  
    
    plt.figure(figsize=(15,8)) 
    title = "MLP regression plot : error="+str(calc_mape)+"%"
    plt.title(title,fontdict={'fontsize': 15})
    plt.plot(x, y_forecast, color='red',label="Predicted data")
    plt.plot(x, y_actual, color='green',label="Actual data")
    plt.xlabel('Days',fontsize=15)
    plt.ylabel("Daily numbers",fontsize=15)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.legend(loc="best")
    img = "static\\plotted_mlp.jpg"
    plt.savefig(img)
    return send_file(img,as_attachment=True,mimetype='image/jpg') 

if __name__ == "__main__":
    app.run(debug=True)