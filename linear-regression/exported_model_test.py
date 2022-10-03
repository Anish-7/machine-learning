import joblib
import pandas as pd
regressor = joblib.load('regressor.joblib')

x_test=pd.read_csv('x_test.csv')
x=x_test.loc[:,['Close','High']]

prediction=regressor.predict(x)
print(prediction)