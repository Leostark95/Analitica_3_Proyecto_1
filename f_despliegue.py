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

    # Leer datos para 2016 
    df_2016 = pd.read_sql("""
        SELECT * FROM processed_data_2016
        """, conn)

    #Preparar los datos para poder realizar las predicciones.
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
    
    #Solo guardar los valores de los empleados que tienen el rol Sales Executive
    rol_critico = perf_pred[perf_pred['JobRole_Sales Executive'] == 1]

    # Exportar los IDs de los empleados y la variable pred_renuncia_2017
    rol_critico[['EmployeeID', 'pred_renuncia_2017']].to_excel(f"salidas/predicciones.xlsx", index=False)

    # Cerrar cursor
    curr.close()
