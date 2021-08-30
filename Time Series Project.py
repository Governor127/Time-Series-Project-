#Improting libraries
import numpy as np
import pandas as pd
import matplotlib.pylot as plt #plot within notebook
import tensorflow as tf
from tensorflow import keras
from keras.model import Sequential
from keras.layer import SimpleRNN, Dense, Input

#Reading data
path = "Uniqlo(FastRetailing) 2012-2016 Training - stocks2012-2016.csv"
df = pd.read_csv(path)
 #Reading the first and last 5 rows of thr dataframe 
df.head(5)
df.tail(5)
#focusing on the hign value of the stock
df = df{"high"}
df head()

#function to plot the time series
def plot_series(data): 
    for df in data:
       plt.plot(df)
plt.show()

#Splitting the train-test data using 80% of the whole dataset for training and tearting 
train = df[:int(0.8*len(df))]
test = df[int(0.8*len(df)):]

#Reshaping the data
def arrange(data, window=10):
x=[]
y=[]
#looping through the data
for i, val in enumerate(data):
    if i < window:
        continue #for insufficient number of past record available, the continue
        x.append((data[i-window: i-1]).values.reshape(-1,1))
        y.append(data[i:i+1].values.reshape(-1,1))
#converting the list into numpy array for fast computation
        x = np.asarray(x)
        y = np.asarray(y)

        return x,y

X_train, y_train = arrange(train, 15)
    print("x-shape is: {} and y-shape is: {}". format(X-train.shape, y_train.shape))
#to demonstrate how a data point looks like
print(X_train[0])
print(y_train[0])

#Building the model
Net = Sequential()
Net.add(SimpleRNN(3, activation='relu', input_shape=X_train.shape[1:]))
Net.summary()
Net.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
Net.fit(X_train, y_train, epochs=10)
#checking the performance of the model
X_test, y_test = arrange(test, 15)
pred = Neet.predict(X_test)
pred[:, 0]
#plotting to campare predictions with the actual value 
plot_series([pred[:, 0], y_test[:, :, 0]])