#----------- Librerías -----------------
import a_funciones as funciones  ###archivo de funciones propias
import pandas as pd ### para manejo de datos
import sqlite3 as sql
import joblib
import openpyxl ## para exportar a excel
import numpy as np


###### el despliegue consiste en dejar todo el código listo para una ejecucion automática en el periodo definido:
###### en este caso se ejecutara el proceso de entrenamiento y prediccion anualmente.
if __name__=="__main__":

#------------ Cargar datos ---------------------
    conn = sql.connect('data/my_database.db')
    curr = conn.cursor()

    curr.close()
    # Leer datos para 2016 
    df_2016 = pd.read_sql("""
        SELECT * FROM processed_data_2016
        """, conn)


 ####Otras transformaciones en python (imputación, dummies y seleccion de variables)

    df_t= funciones.preparar_datos(df_2016)
    df_t


    ##Cargar modelo y predecir
    m_rfc = joblib.load("salidas\\rf_final.pkl")
    predicciones=m_rfc.predict(df_t)
    pd_pred=pd.DataFrame(predicciones, columns=['pred_renuncia_2017'])


    ###Crear base con predicciones ####

    perf_pred=pd.concat([df_2016['EmployeeID'],df_t,pd_pred],axis=1)
   
    ####LLevar a BD para despliegue 
    perf_pred.loc[:,['EmployeeID', 'pred_renuncia_2017']].to_sql("perf_pred",conn,if_exists="replace") ## llevar predicciones a BD con ID Empleados
    

    ####ver_predicciones_bajas ###
    # Filtrar empleados con más del 80% de probabilidad de retirarse
    empleados_riesgo_alto = perf_pred[perf_pred['pred_renuncia_2017']>0.8]

    # Mostrar los empleados con alto riesgo de retiro
    print(empleados_riesgo_alto)
    
    importances = m_rfc.feature_importances_
    columnas = df_t.columns
    coeficientes = pd.DataFrame({'caracteristicas': columnas, 'importancia': importances})
    coeficientes.to_excel("salidas\\importancia_caracteristicas.xlsx", index=False)

    # Exportar predicciones más bajas y variables explicativas
    empleados_riesgo_alto.to_excel("salidas\\prediccion.xlsx", index=False)

'''
importancia_caracteristicas.xlsx
Contenido: Este archivo contiene la importancia de cada una de las variables (características) utilizadas en el modelo de bosque aleatorio (RandomForestClassifier). La importancia indica cuánto contribuye cada característica a las predicciones del modelo.

Columnas:

caracteristicas: El nombre de cada variable utilizada en el modelo.
importancia: Un valor numérico que representa la importancia relativa de cada característica para hacer las predicciones.
Utilidad:

Puedes utilizar este archivo para analizar qué variables son más relevantes para la predicción de la renuncia de los empleados.
Este análisis puede ayudar a enfocar futuros esfuerzos en las variables más importantes y a descartar aquellas que tienen poca influencia en el modelo.
También puede ayudarte a comprender mejor el comportamiento del modelo y justificar decisiones basadas en los resultados obtenidos.
2. prediccion.xlsx
Contenido: Este archivo contiene las predicciones del modelo para los empleados de 2016, junto con su EmployeeID y las características que se usaron en el modelo. También filtra a los empleados con más del 80% de probabilidad de renunciar.

Columnas:

EmployeeID: El identificador único de cada empleado.
pred_renuncia_2017: La predicción del modelo sobre si el empleado renunciará en 2017 (generalmente con 1 indicando "sí" y 0 indicando "no").
Otros: Las columnas correspondientes a las variables utilizadas en el modelo.
Utilidad:

Este archivo te permite identificar a los empleados en riesgo de renunciar según las predicciones del modelo. Puedes enfocarte en aquellos con más del 80% de probabilidad de renunciar.
La empresa puede usar esta información para tomar acciones preventivas, como intervenciones de recursos humanos, mejoras en el ambiente laboral o programas de retención.
También sirve para monitorear la precisión de las predicciones y verificar si coinciden con la realidad.
Acciones que puedes tomar con estos archivos:
Análisis de importancia de características: Usa el archivo importancia_caracteristicas.xlsx para identificar las variables más influyentes y optimizar futuros modelos o intervenciones en la empresa.
Prevención de renuncias: A partir de prediccion.xlsx, puedes focalizar esfuerzos en los empleados de mayor riesgo y tomar decisiones informadas para mejorar la retención del personal.
Revisión del modelo: Con el archivo de predicciones puedes verificar si los resultados son coherentes y ajustar el modelo si es necesario.
'''