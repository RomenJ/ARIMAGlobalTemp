import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Suprimir advertencias de SettingWithCopyWarning
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# Cargar el dataset de temperaturas por país
dataGlobalbyCount = pd.read_csv('GlobalLandTemperaturesByCountry.csv', parse_dates=['dt'], index_col='dt')

# Mostrar los primeros 5 registros del DataFrame
print("Value Counts:")
print(dataGlobalbyCount.head())

# Información general del DataFrame
print("Info:")
dataGlobalbyCount.info()

# Conteo de registros por país
print("Number of Values:")
print(dataGlobalbyCount['Country'].value_counts())

# Filtrar países con alta frecuencia y eliminar filas con valores nulos en AverageTemperature
high_freq_countries = dataGlobalbyCount['Country'].value_counts()[lambda x: x > 3228].index
print(high_freq_countries)

HighFrecCountries = dataGlobalbyCount[dataGlobalbyCount['Country'].isin(high_freq_countries)].dropna(subset=['AverageTemperature'])

# Seleccionar los países de interés
countries_to_add = ['Spain', 'Faroe Islands', 'Iceland', 'Greece', 'Germany', 'Saint Pierre And Miquelon', 'France', 'France (Europe)', 'Finland']
SHighFrecCountries = HighFrecCountries[HighFrecCountries['Country'].isin(countries_to_add)]

# Crear una nueva columna para el siglo
SHighFrecCountries.loc[:, 'Century'] = (SHighFrecCountries.index.year // 100 + 1) * 100

# Función para realizar la prueba ADF y mostrar los resultados
def adf_test(series, country_name):
    result = adfuller(series.dropna())
    print(f'ADF Statistic {country_name}:', result[0])
    print(f'p-value {country_name}:', result[1])
    print(f'Critical Values {country_name}:', result[4])

# Prueba ADF para cada país
def analyze_country(country_name):
    country_temperatures = SHighFrecCountries[SHighFrecCountries['Country'] == country_name]['AverageTemperature']
    adf_test(country_temperatures, country_name)
    plot_and_predict(country_temperatures, country_name)

# Función para resamplear, graficar y predecir temperaturas
def plot_and_predict(country_temperatures, country_name, start_year='1744', end_year='2053'):
    yearly_temperatures = country_temperatures.dropna().resample('A').median().to_frame()

    plt.figure(figsize=(10, 5))
    plt.plot(yearly_temperatures.index, yearly_temperatures['AverageTemperature'])
    plt.title(f'Temperatura Mediana Anual en {country_name}')
    plt.xlabel('Año')
    plt.ylabel('Temperatura Mediana (°C)')
    plt.grid(True)
    plt.savefig(f'{country_name}_annual_median_temperature.png')
    plt.show()

    model = ARIMA(yearly_temperatures, trend='t', order=(1,1,1))
    result = model.fit()

    fig, ax = plt.subplots(figsize=(10, 5))
    yearly_temperatures.plot(ax=ax)
    plot_predict(result, start=start_year, end=end_year, ax=ax)
    plt.title(f'Predicción de Temperatura en {country_name}')
    plt.savefig(f'{country_name}_temperature_prediction.png')
    plt.show()

# Análisis y predicción para cada país
for country in countries_to_add:
    analyze_country(country)

print(SHighFrecCountries)
