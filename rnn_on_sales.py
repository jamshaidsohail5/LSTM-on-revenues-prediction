
# Part 1 - Data Preprocessing

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Importing the training set
dataset = pd.ExcelFile('/home/ubuntu/tmp/Sales.xlsx')
df = dataset.parse('Sales')

# Manipulating the Date
df['Date'] =pd.to_datetime(df.Date)
df =df.sort_values(by=['Date'])
df.dropna(subset=['Revenue'], how='all', inplace = True)


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
df1, df2 = train_test_split(df, test_size = 0.2, random_state = 0)



# Getting the revenue in the program
training_set = df1.iloc[:, 4:5].values
training_set = training_set[~np.isnan(training_set).any(axis=1)]



# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)


# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(100, 838730):
    X_train.append(training_set_scaled[i-100:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))




# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()


# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))


# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))


# Adding the output layer
regressor.add(Dense(units = 1))


# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 5, batch_size = 32)


# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 
real_revenues = df2.iloc[:, 4:5].values


# Getting the predicted stock price of 2017

dataset_total = pd.concat((df1['Revenue'], df2['Revenue']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(df2) - 100:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
to_be_predicted = []

for i in range(100, 209808):
    to_be_predicted.append(inputs[i-100:i, 0])


to_be_predicted = np.array(to_be_predicted)
to_be_predicted = np.reshape(to_be_predicted, (to_be_predicted.shape[0], to_be_predicted.shape[1], 1))
predicted_revenues = regressor.predict(to_be_predicted)
predicted_revenues = sc.inverse_transform(predicted_revenues)



# Visualising the results
plt.plot(real_revenues, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_revenues, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


