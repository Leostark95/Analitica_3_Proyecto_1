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
        SELECT EmployeeID, Age, BusinessTravel, Department, DistanceFromHome,
            Education, JobRole, MonthlyIncome, NumCompaniesWorked,
            PercentSalaryHike, TrainingTimesLastYear, YearsAtCompany,
            YearsSinceLastPromotion, EnvironmentSatisfaction, JobSatisfaction,
            WorkLifeBalance, JobInvolvement, PerformanceRating 
        FROM processed_data_2016
        """, conn)
    v_selc = pd.read_sql("Select * From v_seleccionadas", conn)

 ####Otras transformaciones en python (imputación, dummies y seleccion de variables)
    df_t= funciones.preparar_datos(df_2016,v_selc)


    ##Cargar modelo y predecir
    m_lreg = joblib.load("salidas\\modelo.pkl")
    predicciones=m_lreg.predict(df_t)
    pd_pred=pd.DataFrame(predicciones, columns=['pred_renuncia_2017'])


    ###Crear base con predicciones ####

    perf_pred=pd.concat([df_2016['EmployeeID'],df_t,pd_pred],axis=1)
   
    ####LLevar a BD para despliegue 
    perf_pred.loc[:,['EmployeeID', 'pred_renuncia_2017']].to_sql("perf_pred",conn,if_exists="replace") ## llevar predicciones a BD con ID Empleados
    

    ####ver_predicciones_bajas ###
    # Filtrar empleados con más del 80% de probabilidad de retirarse
    empleados_riesgo_alto = perf_pred[perf_pred['pred_perf_2024'] > 0.80]

    # Mostrar los empleados con alto riesgo de retiro
    print(empleados_riesgo_alto)
    
    coeficientes=pd.DataFrame( np.append(m_lreg.intercept_,m_lreg.coef_) , columns=['coeficientes'])  ### agregar coeficientes
   
    empleados_riesgo_alto.to_excel("salidas\\prediccion.xlsx")   #### exportar predicciones mas bajas y variables explicativas
    coeficientes.to_excel("salidas\\coeficientes.xlsx") ### exportar coeficientes para analizar predicciones
    

