import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# Cargar los archivos CSV o Excel
sales_real_data = pd.read_excel('C:/Users/cjret/OneDrive/Escritorio/Tesis/Modelos/sales_july.xlsx')  # Ventas reales
lstm_predictions_data = pd.read_excel("C:/Users/cjret/OneDrive/Escritorio/Tesis/Modelos/predicciones_turnover_lstm.xlsx")  # Predicciones LSTM
trend_data = pd.read_csv("C:/Users/cjret/OneDrive/Escritorio/Tesis/Modelos/Trend.csv")  # Trend de ventas
gru_predictions_data = pd.read_excel("C:/Users/cjret/OneDrive/Escritorio/Tesis/Modelos/predicciones_turnover_gru.xlsx")  # Predicciones GRU
lstm_exog_predictions_data = pd.read_excel("C:/Users/cjret/OneDrive/Escritorio/Tesis/Modelos/predicciones_turnover_lstm_exog.xlsx")  # Predicciones LSTM Exog
gru_exog_predictions_data = pd.read_excel("C:/Users/cjret/OneDrive/Escritorio/Tesis/Modelos/predicciones_turnover_gru_exog.xlsx")  # Predicciones GRU Exog
sarima_predictions_data = pd.read_excel('C:/Users/cjret/OneDrive/Escritorio/Tesis/Modelos/predicciones_sarima.xlsx')  # Predicciones SARIMA
sarimax_predictions_data = pd.read_excel('C:/Users/cjret/OneDrive/Escritorio/Tesis/Modelos/predicciones_sarimax.xlsx')  # Predicciones SARIMAX

# Convertir las columnas de fechas a formato datetime para asegurar que estén alineadas
sales_real_data['date'] = pd.to_datetime(sales_real_data['date'])
lstm_predictions_data['date'] = pd.to_datetime(lstm_predictions_data['date'])
trend_data['date'] = pd.to_datetime(trend_data['date'])
gru_predictions_data['date'] = pd.to_datetime(gru_predictions_data['date'])
lstm_exog_predictions_data['date'] = pd.to_datetime(lstm_exog_predictions_data['date'])
gru_exog_predictions_data['date'] = pd.to_datetime(gru_exog_predictions_data['date'])
sarima_predictions_data['date'] = pd.to_datetime(sarima_predictions_data['date'])
sarimax_predictions_data['date'] = pd.to_datetime(sarimax_predictions_data['date'])

# Filtrar los datos para los primeros 28 días de julio
date_range = pd.date_range(start="2024-07-01", end="2024-07-28")
sales_real_28 = sales_real_data[sales_real_data['date'].isin(date_range)].reset_index(drop=True)
lstm_predictions_28 = lstm_predictions_data[lstm_predictions_data['date'].isin(date_range)].reset_index(drop=True)
trend_28 = trend_data[trend_data['date'].isin(date_range)].reset_index(drop=True)
gru_predictions_28 = gru_predictions_data[gru_predictions_data['date'].isin(date_range)].reset_index(drop=True)
lstm_exog_predictions_28 = lstm_exog_predictions_data[lstm_exog_predictions_data['date'].isin(date_range)].reset_index(drop=True)
gru_exog_predictions_28 = gru_exog_predictions_data[gru_exog_predictions_data['date'].isin(date_range)].reset_index(drop=True)
sarima_predictions_28 = sarima_predictions_data[sarima_predictions_data['date'].isin(date_range)].reset_index(drop=True)
sarimax_predictions_28 = sarimax_predictions_data[sarimax_predictions_data['date'].isin(date_range)].reset_index(drop=True)

# Extraer los valores de facturación (turnover) para calcular las métricas
y_real = sales_real_28['turnover'].values
y_pred_lstm = lstm_predictions_28['predicted_turnover'].values
y_pred_gru = gru_predictions_28['predicted_turnover'].values
y_pred_trend = trend_28['turnover'].values
y_pred_sarima = sarima_predictions_28['predicted_turnover'].values
y_pred_lstm_exog = lstm_exog_predictions_28['predicted_turnover'].values
y_pred_gru_exog = gru_exog_predictions_28['predicted_turnover'].values
y_pred_sarimax = sarimax_predictions_28['predicted_turnover'].values

# Función para calcular métricas
def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    re = np.mean((y_pred - y_true) / y_true)
    are = np.mean(np.abs(y_pred - y_true) / y_true)
    return rmse, mae, re, are

# Crear un diccionario para almacenar los resultados
metrics = {
    "LSTM": calculate_metrics(y_real, y_pred_lstm),
    "GRU": calculate_metrics(y_real, y_pred_gru),
    "LSTM Exog": calculate_metrics(y_real, y_pred_lstm_exog),
    "GRU Exog": calculate_metrics(y_real, y_pred_gru_exog),
    "SARIMA": calculate_metrics(y_real, y_pred_sarima),
    "SARIMAX": calculate_metrics(y_real, y_pred_sarimax),
    "Trend": calculate_metrics(y_real, y_pred_trend)
}

# Mostrar las métricas para cada modelo
for model, (rmse, mae, re, are) in metrics.items():
    print(f"\nResultados para el modelo {model}:")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"RE: {re}")
    print(f"ARE: {are}")

# Gráfico 1: Comparación entre Trend, SARIMA, GRU y LSTM vs Ventas Reales
plt.figure(figsize=(10, 6))
plt.plot(sales_real_28['date'], y_real, label='Ventas Reales', marker='o')
plt.plot(sales_real_28['date'], y_pred_lstm, label='LSTM', marker='x')
plt.plot(sales_real_28['date'], y_pred_gru, label='GRU', marker='s')
plt.plot(sales_real_28['date'], y_pred_trend, label='Trend', marker='^')
plt.plot(sales_real_28['date'], y_pred_lstm_exog, label='LSTM Exog', marker='D')


# Añadir detalles al gráfico
plt.title('Comparación de Ventas Reales vs LSTM, GRU, Trend, LSTM Exog, GRU Exog')
plt.xlabel('Fecha')
plt.ylabel('Facturación')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%%
# Gráfico 2: Comparación entre SARIMAX, Trend, Ventas Reales, LSTM Exog y GRU Exog
plt.figure(figsize=(10, 6))
plt.plot(sales_real_28['date'], y_real, label='Ventas Reales', marker='o')
plt.plot(sales_real_28['date'], y_pred_lstm_exog, label='LSTM Exog', marker='x')
plt.plot(sales_real_28['date'], y_pred_gru_exog, label='GRU Exog', marker='s')
plt.plot(sales_real_28['date'], y_pred_trend, label='Trend', marker='^')
plt.plot(sales_real_28['date'], y_pred_sarimax, label='SARIMAX', marker='D')

# Añadir detalles al gráfico
plt.title('Comparación de Ventas Reales vs LSTM Exog, GRU Exog, SARIMAX, Trend')
plt.xlabel('Fecha')
plt.ylabel('Facturación')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Gráfico 1: Comparación entre Trend, SARIMA, GRU y LSTM vs Ventas Reales
plt.figure(figsize=(10, 6))
plt.plot(sales_real_28['date'], y_real, label='Ventas Reales', marker='o')
plt.plot(sales_real_28['date'], y_pred_lstm, label='LSTM', marker='x')
plt.plot(sales_real_28['date'], y_pred_trend, label='Trend', marker='x')
#plt.plot(sales_real_28['date'], y_pred_gru_exog, label='GRU Exog', marker='s')
#plt.plot(sales_real_28['date'], y_pred_sarimax, label='SARIMAX', marker='^')



# Añadir detalles al gráfico
#plt.title('Comparación de Ventas Reales vs LSTM, GRU, Trend, LSTM Exog, GRU Exog')
plt.xlabel('Fecha')
plt.ylabel('Turnover')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()