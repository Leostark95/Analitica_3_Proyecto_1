## Importación de librerías

#Librerías para Manipulación de Datos y Operaciones Numéricas
import numpy as np
import pandas as pd

#Librerías para Machine Learning y Preprocesamiento

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel, RFE, VarianceThreshold, SelectKBest, f_regression, mutual_info_regression, f_classif, mutual_info_classif, chi2
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#Librerías para Visualización

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, Markdown

#Otras Librerías

import joblib # Para guardar y cargar objetos Python.
import os # Para operaciones del sistema de archivos.

# Las fucniones más útiles para el desarrollo del proyecto están en este script

## Funciones y descripciones

#--------------------Funciones para manipulación de archivos-------------------------------
#Funció ejecutar_sql para leer 
def ejecutar_sql (nombre_archivo,cur):
  '''
    Esta función ejecuta archivos sql desde python. para ello toma dos parámetros:
    nombre_archivo: El nombre del archivo SQL que contiene el script a ejecutar.
    cur: Un objeto cursor de la conexión a la base de datos, que se utiliza para ejecutar comandos SQL.
  '''
  sql_file=open(nombre_archivo)
  sql_as_string=sql_file.read()
  sql_file.close()
  cur.executescript(sql_as_string)

#La función eliminar_archivos elimina archivos del repositorio si existen
def eliminar_archivos(rutas_archivos):
    """
    Elimina los archivos existentes, esto con el fin de poder realizar pruebas y no realizar el 
    borrado de manera manual, pues la base de datos creada para guardar las tablas de sql no se puede
    sobreescribir y por tanto no se actualizaba, entonces se decidió por implementar esta función para
    hacerse de manera automática.

    Esta función recibe como parámetro la ruta del archivo a eliminar.
    """
    for ruta in rutas_archivos:
        #Se realiza un condicional de si existe o no la ruta para evitar errores
        if os.path.exists(ruta):
            os.remove(ruta)
            print(f"Archivo eliminado: {ruta}")
        else:
            print(f"El archivo no existe: {ruta}")

#--------------------análisis exploratorio-------------------------------
#Función para el análisis descriptivo
def check_df(dataframe, head=5):
    '''
    Esta función imprime un resumen detallado del dataframe, como el número de columnas y filas que tiene,
    el tipo de dato que es, una pequeña observación del dataframe, muestra los nulos de cada variable y 
    si tiene duplicados.
    '''
    display(Markdown('**Shape**'))
    display(dataframe.shape)

    display(Markdown('**Types**'))
    display(dataframe.dtypes)

    display(Markdown('**Head**'))
    display(dataframe.head(head))

    display(Markdown('**NA**'))
    display(dataframe.isnull().sum())

    display(Markdown("**Duplicated**"))
    display(dataframe.duplicated().sum())

#Visualización de datos
#Función para la generación de gráfico histograma y boxplot
def plot_hist_box(df, list_numericas):
    '''
    Esta función recibe como argumentos un dataframe y una lista de nombre de variables
    con el fin de crear un gráfico histograma y otro de boxplot para cada una de estas variables numércias.
    '''
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    color = 'lightblue'
    
    # Histograma
    sns.histplot(df[list_numericas], kde=False, color=color, ax=axs[0])
    axs[0].set_title(f'Histograma de {list_numericas}')
    axs[0].set_xlabel(list_numericas)
    axs[0].set_ylabel('Frecuencia')
    
    # Boxplot
    sns.boxplot(x=df[list_numericas], color=color, ax=axs[1])
    axs[1].set_title(f'Boxplot de {list_numericas}')
    axs[1].set_xlabel(list_numericas)

    plt.tight_layout()
    plt.show()

#Gráfico para la generación de histogramas
def plot_categorical_distribution(df, list_cat):
    '''
    Esta función recibe como argumento un dataframe y una lista de variables categóricas
    con el fin de crear de manera iterativa un histograma para cada una de estas variables en
    la lista.
    '''
    # Calcular la frecuencia de cada categoría
    data = df[list_cat].value_counts().reset_index()
    data.columns = [list_cat, 'Frecuencia']

    # Crear el gráfico de barras
    fig = px.bar(data, x=list_cat, y='Frecuencia',
                 title=f'Distribución de la Variable Categórica: {list_cat}',
                 labels={list_cat: list_cat, 'Frecuencia': 'Frecuencia'},
                 color='Frecuencia',  # dar color a las barras según la frecuencia
                 color_continuous_scale='Viridis')  # escala de colores

    # Ajustar la apariencia del gráfico
    fig.update_layout(template='simple_white', title_x=0.5, showlegend=False)
    fig.update_layout(width=650, height=480)

    # Mostrar el gráfico
    fig.show()

#Función para el gráfico de la variable de respuesta
def plot_renuncia_2016(df, columna_renuncia, titulo='Cantidad de Renuncias en 2016'):
    """
    Genera un gráfico de barras que muestra la cantidad de renuncias en 2016, recibe como argumentos un
    dataframe, el nombre de la variable renuncia y por último un titulo.
    """
    # Contar la cantidad de renuncias
    conteo_renuncia = df[columna_renuncia].value_counts()

    # Definir la paleta de colores
    colores = sns.color_palette('pastel', len(conteo_renuncia))

    # Crear el gráfico de barras
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=conteo_renuncia.index, y=conteo_renuncia.values, palette=colores)

    # Configurar etiquetas y título
    plt.xlabel('Renuncia en 2016')
    plt.ylabel('Cantidad')
    plt.title(titulo)

    # Añadir etiquetas a las barras
    for i, v in enumerate(conteo_renuncia.values):
        ax.text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=9)

    # Mostrar el gráfico
    plt.show()

# Función de Matriz de correlación v_numéricas
def plot_correlation_matrix(df, list_num):
    '''
    Esta función genera una matriz de correlación que puede observarse por medio de un mapa de calor,
    es idea para analizar si existe multicolinealidad entre las variables.
    Recibe como argumentos:
    - df: dataframe con los datos.
    - list_num: lista de los nombres de las variables numéricas
    '''
    # Calcula la matriz de correlación
    matriz_correlacion = df[list_num].corr()

    # Visualizar la matriz de correlación
    plt.figure(figsize=(15, 12), dpi=80)
    sns.heatmap(matriz_correlacion, annot=True, fmt=".2f", linewidths=0.5, linecolor='white', cmap='coolwarm')
    plt.title("Mapa de calor de la matriz de correlación")
    plt.show()

#Gráficos de variables categóricas vs variable de respuesta
def plot_categorical_vs_binary(df, v_respuesta, list_cat):
    '''
    Esta función genera un gráfico de histograma para cada variable categórica vs la variable de respuesta,
    por ello recibe como argumentos:
    - df: un dataframe con los datos.
    - v_respuesta: Nombre de la variable de respuesta.
    - list_cat: Lista de las variables categóricas

    Esto se hace con el fin de observar que comportamientos tiene la variable de respuesta respecto
    a estas variables categóricas.
    '''
    for col in list_cat:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=col, hue=v_respuesta, data=df, palette='viridis')
        plt.title(f'Distribución de {v_respuesta} por {col}')
        plt.xticks(rotation=45, ha="right")
        plt.show()


#--------------------Escalado de variables-------------------------------
#Función para el escalado de los datos
def escalar_datos(v_num):
    '''
    Esta función escala los datos numéricos para ser utilizados en los modelos.
    Recibe como argumento un dataframe con variables numéricas y retorna un dataframe 
    con las variables escaladas con el standarscaler.
    '''
    sc = StandardScaler()
    x_sc = sc.fit_transform(v_num)

    df_x_sc = pd.DataFrame(x_sc, columns=v_num.columns, index=v_num.index)

    return df_x_sc

#--------------------Selección de variables-------------------------------
# Función recursiva de selección de características
def recursive_feature_selection(X,y,model,k):
    '''
    Esta función tiene como objetivo realizar la selección de variables por medio de la función
    RFE, para ello recive 4 argumentos:
    - X: Datos de entrenamiento
    - y: Datos de prueba
    - model: El modelo entrenado para seleccionar las variables
    - k: El número de variables a seleccionar

    Finalmente retorna X_new que es el dataframe que contiene las k variables que más explican el modelo.
    '''
    rfe = RFE(model, n_features_to_select=k, step=1)
    fit = rfe.fit(X, y)
    X_new = fit.support_
    return X_new

#------------------------ Tratamiento de datos -----------------------------
# Función para el tratamiento de datos
def preparar_datos(df):
    '''
    Esta función tiene el fin de preparar los datos para el modelado y poder realizar las 
    predicciones del siguiente año. Para ello recive como argumento df, el dataframe del 
    año presente y con los datos los cuales se realizarán las predicciones.
    Retorna df_final que es el dataframe con las variables escaladas y seleccionadas para el modelado.
    '''
    #### Cargar listas y scaler
    list_num = joblib.load("salidas\\list_num.pkl")  # Lista de variables numéricas
    list_dummi = joblib.load("salidas\\list_dummi.pkl")  # Lista de variables categóricas para dummizar
    var_names = joblib.load("salidas\\var_names.pkl")  # Lista de nombres de variables finales
    scaler = joblib.load("salidas\\scaler.pkl")  # Standarescaler

    # 1. Eliminar filas con datos faltantes
    df = df.dropna()

    # 2. Transformar variables 'EmployeeID' y 'PercentSalaryHike'
    df['EmployeeID'] = df['EmployeeID'].astype(str)  # Convertir 'EmployeeID' a string
    df['PercentSalaryHike'] = df['PercentSalaryHike'] / 100  # Escalar 'PercentSalaryHike' dividiendo por 100

    # 3. Transformar variables categóricas a string para poder aplicar get_dummies
    for col in list_dummi:
        df[col] = df[col].astype(str)

    # 4. Convertir variables categóricas en variables dummy
    df_dummies = df[list_dummi]
    df_dummies = pd.get_dummies(df_dummies)

    # 5. Escalar las variables numéricas
    x_num = df[list_num]  # Seleccionar las columnas numéricas
    x_scaled = scaler.fit_transform(x_num) # Aplicar el escalador preentrenado
    df_scaled = pd.DataFrame(x_scaled, columns=list_num)  # Crear DataFrame escalado con las columnas numéricas

    # 6. Concatenar las variables escaladas y las variables dummizadas
    df_final = pd.concat([df[['EmployeeID']], df_scaled, df_dummies], axis=1)

    # 7. Seleccionar las variables finales especificadas en var_names
    df_final = df_final[var_names]

    return df_final


