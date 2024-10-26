#%% Librerias
import time

# Inicio del temporizador
start_time = time.time()

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import random as python_random
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.callbacks import Callback, EarlyStopping
import seaborn as sns

#%% Establecer una semilla aleatoria para reproducibilidad
np.random.seed(123)
python_random.seed(123)
tf.random.set_seed(123)

#%% Cargar el DataFrame principal
file_path = "sales_ecom_history.csv"
df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'], errors='coerce', format='%Y-%m-%d')
df['turnover'] = df['turnover'].astype(float)
df = df.set_index('date')

#%% Detección y Reemplazo de Outliers usando el enfoque del Boxplot
def detect_and_replace_outliers_with_interpolation(df):
    # Añadir columnas para el mes y el año
    df['month'] = df.index.month
    df['year'] = df.index.year

    def replace_outliers(group):
        # Calcular los cuartiles y el IQR
        Q1 = group['turnover'].quantile(0.25)
        Q3 = group['turnover'].quantile(0.75)
        IQR = Q3 - Q1

        # Determinar los límites de los outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identificar outliers
        outliers = (group['turnover'] < lower_bound) | (group['turnover'] > upper_bound)

        # Reemplazar outliers por NaN
        group.loc[outliers, 'turnover'] = np.nan

        # Interpolar los NaN
        group['turnover'] = group['turnover'].interpolate(method='linear')

        return group

    # Aplicar la función a cada grupo de mes y año
    df = df.groupby(['year', 'month']).apply(replace_outliers)

    # Eliminar las columnas adicionales
    df.drop(['month', 'year'], axis=1, inplace=True)

    return df

# Detectar y reemplazar outliers
df_filtered = detect_and_replace_outliers_with_interpolation(df)

# Resetear el índice para eliminar 'year' y 'month' del índice
df_filtered = df_filtered.reset_index(drop=True).set_index(df.index)

df_filtered = df_filtered.dropna(subset=['turnover'])

# Graficar la serie temporal
plt.figure(figsize=(12, 6))
plt.plot(df_filtered, label='Turnover')
plt.xlabel('Fecha')
plt.ylabel('Turnover')
plt.title('Serie Temporal de Turnover')
plt.legend()
plt.grid(True)
plt.show()

#%% Tratamiento de los datos
df = df.dropna(subset=['turnover'])

# Escalar los datos
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_filtered)

# Función para crear secuencias
def create_sequences(data, window_size):
    X = []
    y = []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])  # La serie temporal principal está en la única columna
    return np.array(X), np.array(y)

WINDOW_SIZE = 28  # Aumentar el tamaño de la ventana
X, y = create_sequences(scaled_data, WINDOW_SIZE)

# Dividir en entrenamiento, validación y prueba
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)
test_size = len(X) - train_size - val_size

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

#%% Modelo 
class LiveLossPlot(Callback):
    def on_train_begin(self, logs=None):
        self.fig, self.ax = plt.subplots()
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.ax.clear()
        self.ax.plot(self.losses, label='Training Loss')
        self.ax.plot(self.val_losses, label='Validation Loss')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        self.ax.legend()
        self.fig.canvas.draw()

# Construir el modelo LSTM
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW_SIZE, 1)),  # La primera capa LSTM con return_sequences=True
    Dropout(0.2),
    LSTM(128),
    Dropout(0.2),
    Dense(50, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))

# Configurar Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
live_plot = LiveLossPlot()

# Entrenamiento del modelo
model.fit(X_train, y_train, epochs=500, validation_data=(X_val, y_val), callbacks=[live_plot, early_stopping], verbose=1)

#%% Predicciones y evaluación
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Invertir la escala
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_train_inv = scaler.inverse_transform(y_pred_train).flatten()
y_pred_test_inv = scaler.inverse_transform(y_pred_test).flatten()

# Calcular métricas de evaluación
train_mse = np.sqrt(mean_squared_error(y_train_inv, y_pred_train_inv))
test_mse = np.sqrt(mean_squared_error(y_test_inv, y_pred_test_inv))
train_mae = mean_absolute_error(y_train_inv, y_pred_train_inv)
test_mae = mean_absolute_error(y_test_inv, y_pred_test_inv)
abs_relative_error = np.mean(np.abs((y_test_inv - y_pred_test_inv) / y_test_inv))
relative_error = np.mean((y_pred_test_inv - y_test_inv) / y_test_inv)
    

print(f"RMSE train: {train_mse}, RMSE test: {test_mse}, MAE train: {train_mae}, MAE test: {test_mae}, Relative Error: {relative_error}, ABS Relative Error: {abs_relative_error}")

# Graficar las predicciones
plt.figure(figsize=(10, 6))
plt.plot(y_test_inv, label='Real')
plt.plot(y_pred_test_inv, label='Predicted')
plt.title('Real vs Predicted Turnover (LSTM)')
plt.xlabel('Time')
plt.ylabel('Turnover')
plt.legend()
plt.show()


#%% Predicciones 

# Definir el número de días a predecir
N_DAYS = 28  # Puedes cambiar este valor según tus necesidades

# Crear secuencia de predicciones futuras
last_window = scaled_data[-WINDOW_SIZE:]

input_sequence = last_window.reshape((1, WINDOW_SIZE, 1))
predictions = []

# Realizar predicciones iterativas para los próximos N_DAYS
for i in range(N_DAYS):
    # Predecir el próximo valor
    next_pred = model.predict(input_sequence)

    # Guardar la predicción
    predictions.append(next_pred[0, 0])

    # Actualizar la secuencia de entrada con la predicción
    input_sequence = np.append(input_sequence[:, 1:, :], [[next_pred[0]]], axis=1)

# Invertir la escala de las predicciones
predictions_inv = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# Crear un DataFrame con las fechas y las predicciones invertidas
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=N_DAYS)
predictions_df = pd.DataFrame({
    'date': future_dates,
    'predicted_turnover': predictions_inv
})

# Guardar las predicciones en un archivo Excel
predictions_df.to_excel('predicciones_turnover_lstm.xlsx', index=False)

# Mostrar las predicciones
for i, pred in enumerate(predictions_inv):
    print(f"Predicción para el día {i+1}: {pred}")

# Graficar las predicciones
plt.figure(figsize=(10, 6))
plt.plot(range(1, N_DAYS + 1), predictions_inv,  marker='o', linestyle='-', color='b',label='Predicted Turnover')
plt.xlabel('Días en el Futuro')
plt.ylabel('Turnover')
plt.legend()
plt.show()


#%%

# Fin del temporizador
end_time = time.time()

# Mostrar el tiempo total de ejecución
execution_time = end_time - start_time
print(f"El código se ejecutó en {execution_time} segundos")
