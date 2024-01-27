import time
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.utils.multiclass import type_of_target
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

start_time = time.time()

df = pd.read_csv('sensor-data.csv')
df['time'] = pd.to_datetime(df['time'])
df['day_of_week'] = df['time'].dt.dayofweek
df['hour_of_day'] = df['time'].dt.hour
numeric_columns = ['power', 'temp', 'humidity', 'light', 'CO2', 'day_of_week', 'hour_of_day']
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
X_train, X_test, y_train, y_test = train_test_split(df[numeric_columns], df['dust'], test_size=0.2, random_state=42)
model = tf.keras.models.Sequential([tf.keras.layers.Dense(128, activation="relu", input_shape=(7,)), tf.keras.layers.Dense(64, activation="relu"), tf.keras.layers.Dense(1)])

model.compile(optimizer="adam", loss="mse")

model.fit(X_train, y_train, epochs=100)

model.evaluate(X_test, y_test)

y_pred = model.predict(X_test)

model = RandomForestRegressor()
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print("Puntuaciones de validación cruzada:", scores)

end_time = time.time()
execution_time = end_time - start_time
print(f"El tiempo de ejecución fue: {execution_time} segundos")

