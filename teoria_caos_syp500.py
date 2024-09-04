import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# ---------------------------------------Manipulación de datos---------------------------------------
# Cargar el dataset
try:
    df = pd.read_csv("SP500.csv", parse_dates=["Date"])
except FileNotFoundError:
    print("El archivo 'SP500.csv' no se encontró.")
    sys.exit()
except Exception as e:
    print(f"Error al cargar el archivo: {e}")
    sys.exit()

# Verificar columnas necesarias
if "Adj Close" not in df.columns or "Date" not in df.columns:
    print("El archivo CSV debe contener las columnas 'Date' y 'Adj Close'.")
    sys.exit()

# Verificar valores faltantes
if df["Adj Close"].isnull().any() or df["Date"].isnull().any():
    print(
        "El archivo CSV contiene valores faltantes en las columnas 'Date' o 'Adj Close'."
    )
    sys.exit()

# Calcular los rendimientos diarios
df["Returns"] = df["Close"].pct_change()

# Inicializar con el último retorno normalizado
x0 = (df["Returns"].iloc[-1] - df["Returns"].min()) / (
    df["Returns"].max() - df["Returns"].min()
)

# ---------------------------------------Parámetros y constantes---------------------------------------
# Parámetros del modelo logístico
r = 3.782  # Parámetro de control
iteraciones = 100
cant_predicciones = 100

# Parámetros de simulación
variaciones = [90, 110, 120]
ruido_factor = 0.005  # Factor de ruido para agregar aleatoriedad


# ---------------------------------------Funciones---------------------------------------
# Función de simulación con ruido
def simular_precio_con_ruido(df, variacion, ruido_factor):
    df["Simulated"] = df["Adj Close"].copy()
    for i in range(1, len(df)):
        factor_variacion = (
            (df.loc[i, "Adj Close"] - df.loc[i - 1, "Adj Close"])
            / df.loc[i - 1, "Adj Close"]
            * (variacion / 100)
        )
        ruido = np.random.normal(0, ruido_factor)
        df.loc[i, "Simulated"] = df.loc[i - 1, "Simulated"] * (
            1 + factor_variacion + ruido
        )
    return df["Simulated"]


# Función del mapa logístico
def mapeo_logistico(x, r):
    return r * x * (1 - x)


# Función de cálculo del coeficiente de Lyapunov
def exponente_lyapunov(x0, r, iteraciones):
    x = x0
    lyap = 0
    for _ in range(iteraciones):
        x_siguiente = mapeo_logistico(x, r)
        lyap += np.log(abs(r - 2 * r * x))
        x = x_siguiente
    return lyap / iteraciones


# ---------------------------------------Simulación y predicción---------------------------------------
# Simulación de la variación diaria del índice
# Aplicar la simulación para todas las variaciones
simulaciones = {}
for variacion in variaciones:
    label = f"Variación: {variacion}%, Ruido: {ruido_factor}"
    df[label] = simular_precio_con_ruido(df, variacion, ruido_factor)
    simulaciones[label] = df[label]

# Filtrar los datos para la gráfica desde el año 2000 hasta el 6 de junio de 2018
start_date = "2000-01-01"
end_date = "2018-06-06"
df_filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].reset_index(
    drop=True
)

# Configurar la gráfica
fig, ax = plt.subplots(figsize=(12, 6))

# Graficar los precios reales
ax.plot(
    df_filtered["Date"],
    df_filtered["Adj Close"],
    label="Precio Real",
    color="blue",
    linestyle="-",
)

# Graficar las simulaciones filtradas por la fecha
for label in simulaciones.keys():
    ax.plot(df_filtered["Date"], df_filtered[label], label=label)
ax.set_title("Simulación del Precio del S&P 500 (2000-2018)")
ax.set_xlabel("Fecha")
ax.set_ylabel("Precio de Cierre Ajustado")
ax.legend()
ax.set_xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))
plt.show()

# Generación de Predicciones y cálculo del Exponente de Lyapunov
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")

# Generar predicciones
predicciones = [x0]
for _ in range(cant_predicciones):
    x_siguiente = mapeo_logistico(predicciones[-1], r)
    predicciones.append(x_siguiente)

# Convertir predicciones a rendimientos y precios
retornos_predichos = (
    np.array(predicciones[1:]) * (df["Returns"].max() - df["Returns"].min())
    + df["Returns"].min()
)
ult_precio = df["Close"].iloc[-1]
precios_predichos = ult_precio * (1 + retornos_predichos).cumprod()

# Crear fechas para las predicciones
ult_fecha = df.index[-100]
fechas_futuras = pd.date_range(
    start=ult_fecha + pd.Timedelta(days=1), periods=cant_predicciones
)

plt.figure(figsize=(120, 60))
plt.plot(df.index[-500:], df["Close"][-500:], label="Datos históricos", color="blue")
plt.plot(fechas_futuras, precios_predichos, label="Predicciones caóticas", color="red")
plt.plot(
    df.index[-100:],
    df["Close"][-100:],
    label="Datos históricos recientes",
    color="green",
)
plt.title("Predicción del S&P 500 usando un modelo caótico")
plt.xlabel("Fecha")
plt.ylabel("Precio de cierre")
plt.legend()
plt.grid(True)
plt.show()

# Calcular el exponente de Lyapunov
lyap = exponente_lyapunov(x0, r, 1000)
print(f"Exponente de Lyapunov: {lyap}")

# Calcular el error entre las predicciones y los precios reales
error = np.mean(np.abs(precios_predichos - df["Close"].iloc[-100:]))
print(f"Error promedio de las predicciones: ${error}")
