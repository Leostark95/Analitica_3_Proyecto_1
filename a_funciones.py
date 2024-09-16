# Importación de librerías

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer # Para imputación
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
import joblib
from sklearn.preprocessing import StandardScaler # Para escalar variables

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from IPython.display import display, Markdown


#Librerías para métodos de filtrado
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression,  f_classif, mutual_info_classif, chi2
from sklearn.preprocessing import MinMaxScaler

#Librerías para
import os

# Las fucniones más útiles para el desarrollo del proyecto están en este script


#---------------------------------------Función para ejecutar archivos .sql --------------------------------------------
def ejecutar_sql (nombre_archivo,cur):
  sql_file=open(nombre_archivo)
  sql_as_string=sql_file.read()
  sql_file.close()
  cur.executescript(sql_as_string)


#---------------------------------------Función para eliminar archivos--------------------------------------------
'''
La función eliminar_archivos borra los archivos que 
'''
def eliminar_archivos(rutas_archivos):
    """
    Elimina los archivos especificados en la lista de rutas si existen.
    
    :param rutas_archivos: Lista de rutas de archivos a eliminar.
    """
    for ruta in rutas_archivos:
        if os.path.exists(ruta):
            os.remove(ruta)
            print(f"Archivo eliminado: {ruta}")
        else:
            print(f"El archivo no existe: {ruta}")


 # Función para imputar variables numéricas
 
def imputar_numericas (df,tipo):

    if str(tipo)=='mean':
        numericas=df.select_dtypes(include=['number']).columns
        imp_mean=SimpleImputer(strategy='mean')
        df[numericas]=imp_mean.fit_transform(df[numericas])
        return df
    
    if str(tipo)=='most_frequent':
        numericas=df.select_dtypes(include=['number']).columns
        imp_mean=SimpleImputer(strategy='most_frequent')
        df[numericas]=imp_mean.fit_transform(df[numericas])
        return df

# Función para imputar variables categóricas y numéricas

def imputar_f (df,list_cat):  
         
                  
    df_c=df[list_cat]

    df_n=df.loc[:,~df.columns.isin(list_cat)]

    imputer_n=SimpleImputer(strategy='median')
    imputer_c=SimpleImputer(strategy='most_frequent')

    imputer_n.fit(df_n)
    imputer_c.fit(df_c)
    imputer_c.get_params()
    imputer_n.get_params()

    X_n=imputer_n.transform(df_n)
    X_c=imputer_c.transform(df_c)


    df_n=pd.DataFrame(X_n,columns=df_n.columns)
    df_c=pd.DataFrame(X_c,columns=df_c.columns)
    df_c.info()
    df_n.info()

    df=pd.concat([df_n, df_c],axis = 1)
    
    return df

# Función para seleccionar modelos

def sel_variables(modelos,X,y,threshold):
    
    var_names_ac=np.array([])
    for modelo in modelos:
        #modelo=modelos[i]
        modelo.fit(X, y)
        sel=SelectFromModel(modelo, prefit = True, threshold=threshold)
        var_names=modelo.feature_names_in_[sel.get_support()]
        var_names_ac=np.append(var_names_ac, var_names)
        var_names_ac=np.unique(var_names_ac)
    
    return var_names_ac

# Función para validar el rendimiento de los modelos

def medir_modelos(modelos,scoring,X,y,cv):

    metric_modelos=pd.DataFrame()
    for modelo in modelos:
        scores=cross_val_score(modelo,X,y,scoring=scoring,cv=cv )
        pdscores=pd.DataFrame(scores)
        metric_modelos=pd.concat([metric_modelos, pdscores],axis=1)
    
    metric_modelos.columns=["logistic_r","reg_lineal","rf_classifier","decision_tree","random_forest","gradient_boosting""sgd_classifier","xgboost_classifier"]
    return metric_modelos

# Convertir el tipo de dato a fecha

def convertir_fecha(dataframe, columna):

    dataframe[columna] = pd.to_datetime(dataframe[columna])

    return dataframe.info()

# RFE para la selección de variables para distintos modelos

def funcion_rfe(modelos,X,y, num_variables, paso):
  resultados = {}
  for modelo in modelos: 
    rfemodelo = RFE(modelo, n_features_to_select = num_variables, step = paso)
    fit = rfemodelo.fit(X,y)
    var_names = fit.get_feature_names_out()
    puntaje = fit.ranking_
    diccionario_importancia = {}
    nombre_modelo = modelo.__class__.__name__

    for i,j in zip(var_names,puntaje):
      diccionario_importancia[i] = j
      resultados[nombre_modelo] = diccionario_importancia
  
  return resultados

# Diagrama de barras para despliegue de resultados 
def histogram(df1, df2,  columna, name1, name2, color1, color2, titulo):

    fig = make_subplots(rows = 1, cols = 2)
    fig.add_trace(
        go.Histogram(x = df1[columna], name = name1, marker_color = color1),
        row = 1, col = 1
    )

    fig.add_trace(
        go.Histogram(x = df2[columna], name = name2, marker_color = color2),
        row = 1, col = 2
    )

    fig.update_layout(
        title_text = titulo,
        template = 'simple_white')
    fig.show();   
    
# Diagrama de lineas para despliegue de resultados
def line(df, columna1, columna2, titulo, xlabel, ylabel): 

    years_1 = df.groupby([columna1])[[columna2]].count().reset_index()

    fig = px.line(years_1, x = columna1, y = columna2)
    fig.update_layout(
        title = titulo,
        xaxis_title = xlabel,
        yaxis_title = ylabel,
        template = 'simple_white',
        title_x = 0.5)
    fig.show()
    
#--------- Función para análisis descriptivo -------------

def check_df(dataframe, head=5):
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


#-------- Histograma o boxplot ------------
def plot_histogram_and_boxplot(df, column_name):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    color = 'lightblue'
    
    # Histograma
    sns.histplot(df[column_name], kde=False, color=color, ax=axs[0])
    axs[0].set_title(f'Histograma de {column_name}')
    axs[0].set_xlabel(column_name)
    axs[0].set_ylabel('Frecuencia')
    
    # Boxplot
    sns.boxplot(x=df[column_name], color=color, ax=axs[1])
    axs[1].set_title(f'Boxplot de {column_name}')
    axs[1].set_xlabel(column_name)

    plt.tight_layout()
    plt.show()

#----------------- Gráfico histograma -----------------
def plot_categorical_distribution(df, column_name):

    # Calcular la frecuencia de cada categoría
    data = df[column_name].value_counts().reset_index()
    data.columns = [column_name, 'Frecuencia']

    # Crear el gráfico de barras
    fig = px.bar(data, x=column_name, y='Frecuencia',
                 title=f'Distribución de la Variable Categórica: {column_name}',
                 labels={column_name: column_name, 'Frecuencia': 'Frecuencia'},
                 color='Frecuencia',  # Opcional: para dar color a las barras según la frecuencia
                 color_continuous_scale='Viridis')  # Opcional: escala de colores

    # Ajustar la apariencia del gráfico
    fig.update_layout(template='simple_white', title_x=0.5, showlegend=False)
    fig.update_layout(width=650, height=480)

    # Mostrar el gráfico
    fig.show()

#Función para el gráfico de la variable de respuesta
def plot_renuncia_2016(df, columna_renuncia, titulo='Cantidad de Renuncias en 2016'):
    """
    Genera un gráfico de barras que muestra la cantidad de renuncias en 2016.

    Parámetros:
    df (pd.DataFrame): DataFrame que contiene los datos.
    columna_renuncia (str): Nombre de la columna que contiene la información de renuncias.
    titulo (str): Título del gráfico (opcional).
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

#Gráficos de variables categóricas vs variable de respuesta
def plot_categorical_vs_binary(df, v_respuesta, categorical_cols):
    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=col, hue=v_respuesta, data=df, palette='viridis')
        plt.title(f'Distribución de {v_respuesta} por {col}')
        plt.show()

# Función de Matriz de correlación v numéricas

def plot_correlation_matrix(df, columns_num):
    # Calcula la matriz de correlación
    matriz_correlacion = df[columns_num].corr()

    # Visualizar la matriz de correlación
    plt.figure(figsize=(15, 12), dpi=80)
    sns.heatmap(matriz_correlacion, annot=True, fmt=".2f", linewidths=0.5, linecolor='white', cmap='coolwarm')
    plt.title("Mapa de calor de la matriz de correlación")
    plt.show()

#----------------- Métodos de filtrado --------------------------

#Función para el escalado de los datos
def escalar_datos(v_num):
    sc = StandardScaler()
    
    x_sc = sc.fit_transform(v_num)

    df_x_sc = pd.DataFrame(x_sc, columns=v_num.columns, index=v_num.index)
    
    return df_x_sc

# Función recursiva de selección de características
def recursive_feature_selection(X,y,model,k): # model=modelo que me va a servir de estimador en este caso de regresión logística
    rfe = RFE(model, n_features_to_select=k, step=1)# step=1 cada cuanto el toma la sucesión de tomar una caracteristica
    fit = rfe.fit(X, y)
    X_new = fit.support_
    print("Num Features: %s" % (fit.n_features_))
    print("Selected Features: %s" % (fit.support_))
    print("Feature Ranking: %s" % (fit.ranking_))

    return X_new

#Función para el tratamiento de datos
def preparar_datos(df, columnas_selec):

    # 1. Eliminar datos faltantes
    df = df.dropna()

    # Transformar EmployeeID y PercentSalaryHike
    df['EmployeeID'] = df['EmployeeID'].astype(str)
    df['PercentSalaryHike'] = df['PercentSalaryHike']/100

    #2. Transformar variables numéricas a categóricas
    list_cat = ['BusinessTravel', 'Department', 'JobRole',
                'JobSatisfaction', 'WorkLifeBalance', 'Education']
    
    for i in list_cat:
        df[i] = df[i].astype(str)
    
   
    
    # 2. Escalar variables numéricas
    list_num = ['Age', 'DistanceFromHome', 'MonthlyIncome', 
              'NumCompaniesWorked', 'TrainingTimesLastYear', 'YearsAtCompany',
              'YearsSinceLastPromotion','PercentSalaryHike']
    
    x_num =df[list_num]
    df_norm = escalar_datos(x_num)

    # 4. Convertir variables categóricas a Dummis
    dummi = pd.get_dummies(df, columns=list_cat)

    # 5. Seleccionar las variables
    df_final = pd.concat([df['EmployeeID'],df_norm,dummi], axis = 1)

    return df