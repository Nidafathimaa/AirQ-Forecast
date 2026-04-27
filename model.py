import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pickle
from tensorflow.keras.losses import MeanSquaredError


day = pd.read_csv(r"C:\Users\nidhafathima\Documents\MAIN_PROJECTS\data\city_day.csv")
data = day.copy()

data = data.dropna()

data = data[["City", "Date", "PM2.5", "PM10", "AQI"]]


data["Date"] = pd.to_datetime(data["Date"])
data["day"] = data["Date"].dt.day
data["month"] = data["Date"].dt.month
data["year"] = data["Date"].dt.year

data = data.drop("Date", axis=1)

le = LabelEncoder()
data["City"] = le.fit_transform(data["City"])


X = data[["City", "day", "month", "year"]]
y = data[["PM2.5", "PM10", "AQI"]]


scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2
)


X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))


model = Sequential()
model.add(LSTM(50, input_shape=(1, 4)))
model.add(Dense(3))


model.compile(optimizer="adam", loss=MeanSquaredError())

model.fit(X_train, y_train, epochs=5, batch_size=32)


model.save("air_model.keras")

pickle.dump(le, open("label.pkl", "wb"))
pickle.dump(scaler_X, open("scaler_X.pkl", "wb"))
pickle.dump(scaler_y, open("scaler_y.pkl", "wb"))

print("Model trained and saved!")