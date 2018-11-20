import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data.csv')



X = dataset.iloc[:,[6,13]].values
y = dataset.iloc[:,12].values


#Splitting Data to Training and Test Set

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#Fitting Multiple Linear Regression To Training Set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting results on Test set
y_pred = regressor.predict(X)


#Predicting a new result 
y_pred = regressor.predict(X)

for x in range(len(y_pred)):
    if(y_pred[x] > 0.9):
        y_pred[x] = 1
    else:
        y_pred[x] = 0

#Visualising the  Decision Tree Regression Model Results
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = X[:,0]
y = X[:,1]
z = y

ax.scatter(x, y, z, c='blue', marker='*')

ax.set_title(' Multiple Regression Model  ')
ax.set_xlabel(' Target Port')
ax.set_ylabel(' Host Address ')
ax.set_zlabel(' Prediction ')

x1 = X[:,0]
y1 = X[:,1]
z1 = regressor.predict(X)
ax.scatter(x1, y1, z1, c='yellow', marker='*')
plt.show()



    

"""#Adding to the main dataset
import csv
rows=[]
fields=[]
with open('data.csv','r') as csv_input:
    csvreader= csv.reader(csv_input)
    fields=next(csvreader)
    for row in csvreader:
        rows.append(row)
fields.append("Prediction")
i=0        
for row in rows:
    row.append(y_pred[i])
    i+=1        

with open('data.csv','w') as csvfile:
    csvwriter=csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(rows)"""
    
