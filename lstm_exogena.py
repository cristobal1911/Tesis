#%% Librerias

import time

# Inicio del temporizador
start_time = time.time()

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import random as python_random
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.callbacks import Callback

#%% Establecer una semilla aleatoria para reproducibilidad
np.random.seed(123)
python_random.seed(123)
tf.random.set_seed(123)

#%% Cargar el DataFrame

# Leer el archivo CSV
df = pd.read_csv("C:/Users/cjret/OneDrive/Escritorio/Tesis/Modelos/sales_ecom_history.csv")
df['date'] = pd.to_datetime(df['date'], errors='coerce', format='%Y-%m-%d')
df['turnover'] = pd.to_numeric(df['turnover'], errors='coerce')
df = df.set_index('date')

# Comprobar y eliminar filas con fechas inválidas
df = df.dropna(subset=['turnover'])

# Añadir columnas para el mes, el año y el día de la semana
df['month'] = df.index.month
df['year'] = df.index.year
df['day_of_week'] = df.index.dayofweek  # 0 = Lunes, 6 = Domingo
df['month_year'] = df.index.to_period('M')

# Cargar datos exógenos
exog_data = pd.read_csv("C:/Users/cjret/OneDrive/Escritorio/Tesis/Modelos/exogenas_forecast.csv")
exog_data['date'] = pd.to_datetime(exog_data['date'], errors='coerce', format='%Y-%m-%d')
exog_data = exog_data.set_index('date')
df['ciber_day'] = exog_data['ciber_day']

# Guardar una copia de la serie original
original_df = df.copy()

#%% Identificar outliers y reemplazar con interpolación lineal

def replace_outliers_with_interpolation(group):
    # Calcular los cuartiles y el IQR solo para los días que no son ciber_day
    Q1 = group.loc[group['ciber_day'] == 0, 'turnover'].quantile(0.25)
    Q3 = group.loc[group['ciber_day'] == 0, 'turnover'].quantile(0.75)
    IQR = Q3 - Q1

    # Determinar los límites de los outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identificar outliers solo donde ciber_day es 0
    outliers = (group['turnover'] < lower_bound) | (group['turnover'] > upper_bound)
    outliers = outliers & (group['ciber_day'] == 0)

    # Reemplazar outliers por NaN
    group.loc[outliers, 'turnover'] = np.nan

    # Interpolar los NaN
    group['turnover'] = group['turnover'].interpolate(method='linear')

    return group

# Aplicar la función a cada grupo de mes
df = df.groupby(['year', 'month']).apply(replace_outliers_with_interpolation)

# Resetear el índice para eliminar 'year' y 'month' del índice
df = df.reset_index(drop=True).set_index(original_df.index)

# Graficar la serie original y la serie con outliers reemplazados

plt.figure(figsize=(15, 6))
plt.plot(original_df.index, original_df['turnover'], label='Serie Original', color='blue')
plt.title('Serie Temporal Original con Outliers')
plt.xlabel('Fecha')
plt.ylabel('Turnover')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(15, 6))
plt.plot(df.index, df['turnover'], label='Serie con Outliers Interpolados', color='green')

plt.xlabel('Fecha')
plt.ylabel('Turnover')
plt.grid(True)
plt.legend()
plt.show()

#%% Tratamiento de los Datos
df = df.dropna(subset=['turnover'])

df_1 = df[['turnover', 'ciber_day']]

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_1)

# Función para crear secuencias
def create_sequences(data, window_size):
    X = []
    y = []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size, 0])  # Supongamos que la serie temporal principal está en la primera columna
    return np.array(X), np.array(y)

WINDOW_SIZE = 28
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
        self.ax.set_title('Training and Validation Loss')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        self.ax.legend()
        self.fig.canvas.draw()

# Construir el modelo LSTM mejorado
model = Sequential([
    Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(WINDOW_SIZE, X_train.shape[2])),
    MaxPooling1D(pool_size=2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(50, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))

# Configurar Early Stopping
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
live_plot = LiveLossPlot()

# Entrenamiento del modelo
model.fit(X_train, y_train, epochs=300, validation_data=(X_val, y_val), callbacks=[live_plot, early_stopping], verbose=1)

#%% Predicciones y evaluación
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Invertir la escala
y_train_inv = scaler.inverse_transform(np.concatenate((y_train.reshape(-1, 1), X_train[:, -1, 1:]), axis=1))[:, 0]
y_test_inv = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), X_test[:, -1, 1:]), axis=1))[:, 0]
y_pred_train_inv = scaler.inverse_transform(np.concatenate((y_pred_train, X_train[:, -1, 1:]), axis=1))[:, 0]
y_pred_test_inv = scaler.inverse_transform(np.concatenate((y_pred_test, X_test[:, -1, 1:]), axis=1))[:, 0]

# Calcular métricas de evaluación
train_mse = np.sqrt(mean_squared_error(y_train_inv, y_pred_train_inv))
test_mse =  np.sqrt(mean_squared_error(y_test_inv, y_pred_test_inv))
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



#%% Predicciones con variables exógenas

# Definir el número de días a predecir
N_DAYS = 28  # Puedes cambiar este valor según tus necesidades

# Cargar los datos exógenos para los primeros 28 días de julio
exog = pd.read_excel("C:/Users/cjret/OneDrive/Escritorio/Tesis/Modelos/exogenas_july.xlsx")
exog['date'] = pd.to_datetime(exog['date'])
exog = exog.set_index('date')

# Crear secuencia de predicciones futuras (turnover + exógena)
last_window = scaled_data[-WINDOW_SIZE:]

# Asegurarse de que la variable exógena esté disponible para los próximos N_DAYS
exog_future = exog.iloc[:N_DAYS]  # Tomar los primeros 28 días de datos exógenos

# Comprobar si faltan valores en exog_future y llenarlos si es necesario
exog_future = exog_future.fillna(0)  # Puedes ajustar este valor si faltan datos

# Crear secuencia de entrada inicial con las variables exógenas incluidas
input_sequence = last_window.reshape((1, WINDOW_SIZE, 2))  # Asumimos que tienes 2 variables: turnover y ciber_day
predictions = []

# Realizar predicciones iterativas para los próximos N_DAYS
for i in range(N_DAYS):
    # Predecir el próximo valor
    next_pred = model.predict(input_sequence)

    # Guardar la predicción
    predictions.append(next_pred[0, 0])

    # Actualizar la secuencia de entrada con la predicción y el valor de ciber_day futuro
    # next_pred[0] es un solo valor y exog_future.iloc[i, 0] es otro, necesitamos unificarlos
    next_input = np.array([[next_pred[0, 0], exog_future.iloc[i, 0]]])  # Crear un array de (1, 2)
    
    # Actualizar la secuencia de entrada (eliminamos la primera ventana y añadimos la nueva)
    input_sequence = np.append(input_sequence[:, 1:, :], next_input.reshape(1, 1, 2), axis=1)

# Invertir la escala de las predicciones
predictions_inv = scaler.inverse_transform(np.concatenate([np.array(predictions).reshape(-1, 1), exog_future.values], axis=1))[:, 0]

# Crear un DataFrame con las fechas y las predicciones invertidas
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=N_DAYS)
predictions_df = pd.DataFrame({
    'date': future_dates,
    'predicted_turnover': predictions_inv
})

# Guardar las predicciones en un archivo Excel
predictions_df.to_excel('predicciones_turnover_gru_exog.xlsx', index=False)

# Mostrar las predicciones
for i, pred in enumerate(predictions_inv):
    print(f"Predicción para el día {i+1}: {pred}")

# Graficar las predicciones
plt.figure(figsize=(10, 6))
plt.plot(range(1, N_DAYS + 1), predictions_inv, marker='o', linestyle='-', color='b', label='Predicted Turnover')
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
