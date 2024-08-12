import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller

#%% Cargar el DataFrame

# Leer el archivo CSV
df = pd.read_csv("sales_ecom_history.csv")
df['date'] = pd.to_datetime(df['date'], errors='coerce', format='%Y-%m-%d')
df['turnover'] = pd.to_numeric(df['turnover'], errors='coerce')  # Convertir a numérico, NaN en caso de error
df = df.set_index('date')

#%%
# Comprobar y eliminar filas con fechas inválidas
df = df.dropna(subset=['turnover'])

# Añadir columnas para el mes, el año y el día de la semana
df['month'] = df.index.month
df['year'] = df.index.year
df['day_of_week'] = df.index.dayofweek  # 0 = Lunes, 6 = Domingo
df['month_year'] = df.index.to_period('M')

# Guardar una copia de la serie original
original_df = df.copy()

#%% Identificar outliers y reemplazar con interpolación lineal
def replace_outliers_with_interpolation(group):
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

# Aplicar la función a cada grupo de mes
df = df.groupby(['year', 'month']).apply(replace_outliers_with_interpolation)

# Resetear el índice para eliminar 'year' y 'month' del índice
df = df.reset_index(drop=True).set_index(original_df.index)

# Eliminar las columnas adicionales y mantener solo date y turnover
df = df[['turnover']]

#%% Graficar la serie original y la serie con outliers reemplazados

# Graficar la serie original
plt.figure(figsize=(15, 6))
plt.plot(original_df.index, original_df['turnover'], label='Serie Original', color='blue')
plt.title('Serie Temporal Original con Outliers')
plt.xlabel('Fecha')
plt.ylabel('Turnover')
plt.grid(True)
plt.legend()
plt.show()

# Graficar la serie después de reemplazar los outliers
plt.figure(figsize=(15, 6))
plt.plot(df.index, df['turnover'], label='Serie con Outliers Interpolados', color='green')
plt.title('Serie Temporal con Outliers Reemplazados por Interpolación')
plt.xlabel('Fecha')
plt.ylabel('Turnover')
plt.grid(True)
plt.legend()
plt.show()

#%% Prueba de Estacionariedad

# Comprobar y eliminar filas con turnover inválido
df = df.dropna(subset=['turnover'])

# Realizar la prueba de Dickey-Fuller Aumentada
result = adfuller(df['turnover'])

# Mostrar los resultados de la prueba
print('Resultados de la Prueba Dickey-Fuller Aumentada:')
print(f'Estadístico ADF: {result[0]:.4f}')
print(f'Valor p: {result[1]:.4f}')
print('Valores Críticos:')
for key, value in result[4].items():
    print(f'   {key}: {value:.4f}')
    
#%% Dividir los datos en conjuntos de entrenamiento y prueba

# Utilizar los últimos 28 días para prueba
test_size = 28
train, test = df['turnover'].iloc[:-test_size], df['turnover'].iloc[-test_size:]

#%% Identificar los mejores parámetros para SARIMA

# Usar auto_arima para identificar los parámetros óptimos
auto_arima_model = auto_arima(train, 
                              seasonal=True, 
                              m=28,  # Ajustar según el análisis de patrones estacionales
                              trace=True, 
                              error_action='ignore', 
                              suppress_warnings=True, 
                              stepwise=True)

print(auto_arima_model.summary())

#%% Transformación Logarítmica
# Aplicar transformación logarítmica para estabilizar la varianza, si la serie es positiva
df['turnover_log'] = np.log(df['turnover'])

# Comprobar y eliminar filas con turnover inválido
df = df.dropna(subset=['turnover_log'])

#%% Dividir los datos en conjuntos de entrenamiento y prueba

# Utilizar los últimos 28 días para prueba
test_size = 28
train, test = df['turnover_log'].iloc[:-test_size], df['turnover_log'].iloc[-test_size:]

#%% Identificar los mejores parámetros para SARIMA

# Usar auto_arima para identificar los parámetros óptimos
auto_arima_model = auto_arima(train, 
                              seasonal=True, 
                              m=28,  # Ajustar según el análisis de patrones estacionales
                              trace=True, 
                              error_action='ignore', 
                              suppress_warnings=True, 
                              stepwise=True)

print(auto_arima_model.summary())

#%% Ajustar el Modelo SARIMA

# Obtener los mejores parámetros del modelo
order = auto_arima_model.order
seasonal_order = auto_arima_model.seasonal_order

# Ajustar el modelo SARIMA
sarima_model = SARIMAX(train, 
                       order=order, 
                       seasonal_order=seasonal_order,
                       enforce_stationarity=False,
                       enforce_invertibility=False)

sarima_fit = sarima_model.fit(disp=False)

# Predicciones en el conjunto de prueba
sarima_pred_log = sarima_fit.forecast(len(test))

# Convertir las predicciones de nuevo a la escala original
sarima_pred = np.exp(sarima_pred_log)

index_values = test.index

sarima_pred.index = index_values

# Evaluar el modelo
rmse = np.sqrt(mean_squared_error(np.exp(test), sarima_pred))
mae = mean_absolute_error(np.exp(test), sarima_pred)
abs_relative_error = np.mean(np.abs((np.exp(test) - sarima_pred) / np.exp(test)))
relative_error = np.mean(((np.exp(test) - sarima_pred) / np.exp(test)))

print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')

#%% Graficar las Predicciones del Modelo SARIMA

plt.figure(figsize=(15, 6))
plt.plot(train.index, np.exp(train) - 1, label='Entrenamiento', color='blue')
plt.plot(test.index, np.exp(test) - 1, label='Test', color='orange')
plt.plot(test.index, sarima_pred, label='Predicción SARIMA', color='green')
plt.title('Predicciones del Modelo SARIMA (Últimos 28 Días)')
plt.xlabel('Fecha')
plt.ylabel('Turnover')
plt.legend()
plt.grid(True)
plt.show()
