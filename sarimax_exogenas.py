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
exog_data = pd.read_csv("exogenas_forecast.csv")
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

#%% Graficar la serie original y la serie con outliers reemplazados

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
plt.title('Serie Temporal con Outliers Reemplazados por Interpolación')
plt.xlabel('Fecha')
plt.ylabel('Turnover')
plt.grid(True)
plt.legend()
plt.show()

#%% Prueba de Estacionariedad

df = df.dropna(subset=['turnover'])

result = adfuller(df['turnover'])
print('Resultados de la Prueba Dickey-Fuller Aumentada:')
print(f'Estadístico ADF: {result[0]:.4f}')
print(f'Valor p: {result[1]:.4f}')
print('Valores Críticos:')
for key, value in result[4].items():
    print(f'   {key}: {value:.4f}')

#%% Transformación Logarítmica

df['turnover_log'] = np.log(df['turnover'])

df = df.dropna(subset=['turnover_log'])

#%% Dividir los datos en conjuntos de entrenamiento y prueba

test_size = 28
train, test = df['turnover_log'].iloc[:-test_size], df['turnover_log'].iloc[-test_size:]

# Dividir los datos exógenos de manera correspondiente
train_exog, test_exog = df['ciber_day'].iloc[:-test_size], df['ciber_day'].iloc[-test_size:]

#%% Identificar los mejores parámetros para SARIMA

auto_arima_model = auto_arima(train, 
                              exogenous=train_exog,
                              seasonal=True, 
                              m=28,  # Ajustar según el análisis de patrones estacionales
                              trace=True, 
                              error_action='ignore', 
                              suppress_warnings=True, 
                              stepwise=True)

print(auto_arima_model.summary())

#%% Ajustar el Modelo SARIMA

order = auto_arima_model.order
seasonal_order = auto_arima_model.seasonal_order

sarima_model = SARIMAX(train, 
                       exog=train_exog,
                       order=order, 
                       seasonal_order=seasonal_order,
                       enforce_stationarity=False,
                       enforce_invertibility=False)

sarima_fit = sarima_model.fit(disp=False)

sarima_pred_log = sarima_fit.forecast(len(test), exog=test_exog)

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
print(f'Error Relativo Absoluto: {abs_relative_error:.2%}')
print(f'Error Relativo: {relative_error:.2%}')

#%% Graficar las Predicciones del Modelo SARIMA

plt.figure(figsize=(15, 6))
plt.plot(train.index, np.exp(train), label='Entrenamiento', color='blue')
plt.plot(test.index, np.exp(test), label='Test', color='orange')
plt.plot(test.index, sarima_pred, label='Predicción SARIMA', color='green')
plt.title('Predicciones del Modelo SARIMA con Variable Exógena (Últimos 28 Días)')
plt.xlabel('Fecha')
plt.ylabel('Turnover')
plt.legend()
plt.grid(True)
plt.show()
