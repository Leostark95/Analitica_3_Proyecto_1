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

# Función que permite ejecutar un archivo  con extensión .sql que contenga varias consultas

def ejecutar_sql (nombre_archivo,cur):
  sql_file=open(nombre_archivo)
  sql_as_string=sql_file.read()
  sql_file.close()
  cur.executescript(sql_as_string)

# Función para consultar y exportar tablas SQL
def exportar_tabla(nombre_tabla, conn):
    query = f"SELECT * FROM {nombre_tabla}"
    df = pd.read_sql_query(query, conn)
    df.to_csv(f'{nombre_tabla}.csv', index=False)
    print(f"Tabla {nombre_tabla} exportada exitosamente a CSV.")

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


# Cargar y procesar nuevos datos (Transformación)

# Cargar y procesar nuevos datos (Transformación)
def preparar_datos (df):

    # Cargar modelo y listas
    list_cat = joblib.load('Salidas/list_cat.pkl')
    list_dummies = joblib.load('Salidas/list_dummies.pkl')
    var_names = joblib.load('Salidas/var_names.pkl')
    scaler = joblib.load( 'Salidas/scaler.pkl') 

    # Recategorización de variables
    clasificador_education(df, 'EducationField')
    clasificador_jobrole(df,'JobRole')
    df.drop(['EducationField','JobRole'], axis = 1, inplace = True)

    # Ejecutar funciones de transformaciones
    df = imputar_f(df, list_cat)

    df_dummies = pd.get_dummies(df, columns = list_dummies, dtype = int)
    df_dummies = df_dummies.loc[:,~df_dummies.columns.isin(['EmployeeID'])]

    # Ordenamos las variables en el orden de entrenamiento del escalar
    df_dummies = df_dummies.reindex(['Age', 'DistanceFromHome', 'Education', 'JobLevel', 'MonthlyIncome',
       'NumCompaniesWorked', 'PercentSalaryHike', 'StockOptionLevel',
       'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany',
       'YearsSinceLastPromotion', 'YearsWithCurrManager',
       'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance',
       'JobInvolvement', 'PerformanceRating', 'BusinessTravel_Non-Travel',
       'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
       'Department_Human Resources', 'Department_Research & Development',
       'Department_Sales', 'Gender_Female', 'Gender_Male',
       'MaritalStatus_Divorced', 'MaritalStatus_Married',
       'MaritalStatus_Single', 'education_sector_Human Resources',
       'education_sector_Research','education_sector_Marketing',
       'job_rol_Research & Development', 'job_rol_Human Resources',
       'job_rol_Manager', 'job_rol_Sales'], axis = 1)

    X2 = scaler.transform(df_dummies)
    X = pd.DataFrame(X2, columns = df_dummies.columns)
    X = X[var_names]
    
    return X

# Convertir el tipo de dato a fecha

def convertir_fecha(dataframe, columna):

    dataframe[columna] = pd.to_datetime(dataframe[columna])

    return dataframe.info()

# Recategorización de variables por departamentos dado el Rol de trabajo

def clasificador_jobrole(df, nombre_columna):
    df[nombre_columna] = df[nombre_columna].astype('category')

    # Definimos las categorías y cómo las vamos a recategorizar 
    diccionario_rol = {
        'Healthcare Representative': 'Research & Development',
        'Research Scientist': 'Research & Development',
        'Sales Executive': 'Sales',
        'Human Resources': 'Human Resources',
        'Research Director': 'Research & Development',
        'Laboratory Technician': 'Research & Development',
        'Manufacturing Director': 'Research & Development',
        'Sales Representative': 'Sales',
        'Manager': 'Manager'
    }

    # Creamos una columna nueva que contenga la recategorización 
    df["job_rol"] = df[nombre_columna].replace(diccionario_rol)

    return df

# Recategorización de variables por departamentos dado la educación
def clasificador_education(df, nombre_columna):
    df[nombre_columna] = df[nombre_columna].astype('category')

    # Definimos las categorías y cómo las vamos a recategorizar 
    diccionario_educacion = {
        'Life Sciences': 'Research',
        'Other': 'Research',
        'Medical': 'Research',
        'Technical Degree': 'Research',
        'Marketing': 'Marketing',
        'Human Resources': 'Human Resources',
    }

    # Creamos una columna nueva que contenga la recategorización 
    df["education_sector"] = df[nombre_columna].replace(diccionario_educacion)

    return df

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
    
# Resumen de tabla sobre calidad de vida
def table(df1, df2):

    resumen = {
        'EnviromentSatistaction': [df1['EnvironmentSatisfaction'].mode()[0],df2['EnvironmentSatisfaction'].mode()[0]],
        'JobSatisfaction': [df1['JobSatisfaction'].mode()[0],df2['JobSatisfaction'].mode()[0]],
        'WorkLifeBalance': [ df1['WorkLifeBalance'].mode()[0],df2['WorkLifeBalance'].mode()[0]]
    }

    abstract = pd.DataFrame(resumen, index = ['Renuncian', 'No renuncian'])

    return abstract

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

#----------------- Matriz de correlación v numéricas -----------------

def plot_correlation_matrix(df, columns_num):
    # Calcula la matriz de correlación
    matriz_correlacion = df[columns_num].corr()

    # Visualizar la matriz de correlación
    plt.figure(figsize=(15, 12), dpi=80)
    sns.heatmap(matriz_correlacion, annot=True, fmt=".2f", linewidths=0.5, linecolor='white', cmap='coolwarm')
    plt.title("Mapa de calor de la matriz de correlación")
    plt.show()

#----------------- Métodos de filtrado --------------------------

def normalize_dataframe(df):
    # Crear una copia del DataFrame original
    df1 = df.copy(deep=True)
    
    # Asignar el tipo de normalización
    scaler = MinMaxScaler()
    sv = scaler.fit_transform(df1.iloc[:, :])
    
    # Asignar los nuevos datos al DataFrame
    df1.iloc[:, :] = sv
    # Retornar el DataFrame normalizado
    return df1

#------------- Función para escalar -----------------------------

import pandas as pd
from sklearn.preprocessing import StandardScaler

def escalar_datos(v_num):
  
    scaler = StandardScaler()
    

    v_num_esc = scaler.fit_transform(v_num)
    

    v_num_esc = pd.DataFrame(v_num_esc, columns=v_num.columns, index=v_num.index)
    
    return v_num_esc
