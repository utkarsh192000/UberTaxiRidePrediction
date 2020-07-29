import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data=pd.read_csv('taxi.csv')
# print(data.head(5))

data_x=data.iloc[:,0:-1].values    # all rows and all column except the last one which will be excluded
data_y=data.iloc[:,-1].values     # all rows but only last column
# print(data_y)

X_train,X_test,y_train,y_test=train_test_split(data_x,data_y,test_size=0.3,random_state=0)


reg=LinearRegression()
reg.fit(X_train,y_train)

# print("Train score:",reg.score(X_train,y_train))
# print("Test score:",reg.score(X_test,y_test))


pickle.dump(reg,open('taxi.pkl','wb'))



model=pickle.load(open('taxi.pkl','rb'))
print(model.predict([[80,1770000,6000,85]]))