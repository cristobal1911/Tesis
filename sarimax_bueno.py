# -*- coding: utf-8 -*-
"""
Created on Thu May 30 18:39:39 2024

@author: cretamalp
"""

# LIBRERIAS
import pandas as pd
import pmdarima as pmd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
from joblib import Parallel, delayed
import time
import numpy as np
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("C:/Users/cretamalp/Documents/Python Scripts/tesis/date_item_quantity_ecom.csv")
df['date'] = pd.to_datetime(df['date'], errors='coerce', format='%Y-%m-%d')
df['item_number'] = df['item_number'].astype(int)
df['quantity'] = df['quantity'].astype(int)
df = df.set_index('date')

# Ordenando el DataFrame por índice
df = df.sort_index()

# Obteniendo el rango completo de fechas
min_date = df.index.min()
max_date = df.index.max()
all_dates = pd.date_range(start=min_date, end=max_date, freq='D')

# Creando un DataFrame que incluye todas las combinaciones de item_number y las fechas
df_expanded = pd.MultiIndex.from_product([df['item_number'].unique(), all_dates], names=['item_number', 'date']).to_frame(index=False)
df_expanded = df_expanded.set_index(['date', 'item_number'])

# Uniendo el DataFrame original con el expandido para rellenar los valores faltantes
df = df.reset_index().set_index(['date', 'item_number'])
df_complete = df_expanded.join(df, how='left')
df_complete['quantity'] = df_complete['quantity'].fillna(0).astype(int)  # Rellenando con 0 y asegurando tipo int

# Reorganizando el índice si necesario
df_complete = df_complete.reset_index().set_index('date')

df = df_complete


def find_best_m(time_series, max_m):
    best_aic = np.inf
    best_m = None
    results = {}
    
    # Prueba varios valores de m desde 1 hasta max_m
    for m in range(1, max_m + 1):
        try:
            model = auto_arima(time_series, start_p=1, start_q=1,
                               test='adf',  # Prueba de Dickey-Fuller Aumentada
                               seasonal=True, m=m, trace=False,
                               error_action='ignore',  
                               suppress_warnings=True, 
                               stepwise=True)
            aic = model.aic()
            results[m] = aic
            if aic < best_aic:
                best_aic = aic
                best_m = m
        except Exception as e:
            print(f"Error with m={m}: {str(e)}")
            continue
    
    return best_m, results


# Crear un DataFrame para almacenar las predicciones y las métricas
results = []

# Iterar sobre los items en la tienda
for item_number in df['item_number'].unique():
    item = df[df['item_number'] == item_number]
    ts = item[['quantity']]
    
    # Dividir en entrenamiento y prueba
    train_size = int(len(ts) * 0.8)
    train, test = ts[:train_size], ts[train_size:]
    
    # Determinar el mejor m
   #best_m, m_results = find_best_m(train['quantity'], 52)
    #print(f"Best m for item {item_number}: {best_m}")

    # Ajustar el modelo SARIMA con variable exógena
    model = pmd.auto_arima(train['quantity'], 
                           start_p=1, start_q=1, test='adf', m=7, seasonal=True, trace=True)
    
    # Ajustar el modelo SARIMAX con Statsmodels
    sarima = SARIMAX(train['quantity'],  
                     order=model.order, seasonal_order=model.seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False)
    sarima_fitted = sarima.fit(disp=False)
    
    # Hacer predicciones en el conjunto de prueba
    predictions = sarima_fitted.get_forecast(steps=len(test))
    predicted_mean = predictions.predicted_mean

    # Calcular las métricas
    mae = mean_absolute_error(test['quantity'], predicted_mean)
    mse = mean_squared_error(test['quantity'], predicted_mean)
    r2 = r2_score(test['quantity'], predicted_mean)
    
    # Agregar los resultados al DataFrame
    results.append({
        'item_number': item_number,
        'mae': mae,
        'mse': mse,
        'r2': r2
    })

# Convertir los resultados a un DataFrame y mostrarlo
results_df = pd.DataFrame(results)
print(results_df)