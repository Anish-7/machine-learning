from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


try:
    info_path='/home/anish/Documents/code/ml/machine_learning/dataset'
    df=pd.read_csv('/home/anish/Documents/code/ml/machine_learning/dataset/WIKI-AAPL.csv')
except:
    print('file not found')


x=df.loc[:,['Close','High']]

y=df.loc[:,'Open']


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

regressor =LinearRegression()
regressor.fit(x_train,y_train)
y_test=list(y_test)

# filename = "regressor.joblib"
# joblib.dump(regressor, filename)

prediction=regressor.predict(x_test)


for i in range(len(prediction)):
    print(f'{y_test[i]} : prediction : {prediction[i]}')



# viz_train = plt
# viz_train.scatter(x_test['Close'],list(y_test ),color='red')
# viz_train.plot(regressor.predict(x_test), color='blue')
# viz_train.show()
