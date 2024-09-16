import sqlite3
import pandas as pd
import a_funciones as funciones # Este archivo contiene las funciones a utilizar
import sys

# Agregar directorio al path
sys.path.append('C:/Users/delva/OneDrive - Universidad de Antioquia/SEMESTRE 2024-2/ANALITICA 3/Proyecto_Recursos_humanos/Analitica_3_Proyecto_RH')

# Leer los archivos CSV con pandas
general_data = pd.read_csv('data/general_data.csv')
employee_survey_data = pd.read_csv('data/employee_survey_data.csv')
manager_survey_data = pd.read_csv('data/manager_survey.csv')
retirement_info = pd.read_csv('data/retirement_info.csv')

# Mostrar primeras 10 filas
print(general_data.head(10))
print(employee_survey_data.head(10))
print(manager_survey_data.head(10))
print(retirement_info.head(10))

# Vista previa de las bases de datos
columns = general_data.columns
for i in columns:
    print('-'*10, i, '-'*10)
    print(general_data[i].value_counts())

# Eliminar variables que no son relevantes
general_data = general_data.drop(
    [
        'EducationField', 'EmployeeCount', 'MaritalStatus', 'Over18', 
        'StandardHours', 'StockOptionLevel', 'YearsWithCurrManager', 
        'TotalWorkingYears'
    ], 
    axis=1
)

retirement_info = retirement_info.drop(['Attrition', 'resignationReason'], axis = 1)

# Verificar y eliminar nulos
print("Nulos en general_data:")
general_data.isnull().sum()

print("Nulos en employee_survey_data:")
employee_survey_data.isnull().sum()
      
print("Nulos en manager_survey_data:")
manager_survey_data.isnull().sum()

print("Nulos en retirement_info:")
retirement_info.isnull().sum()

# Eliminar filas con valores nulos
general_data = general_data.dropna()
employee_survey_data = employee_survey_data.dropna()

# Eliminar archivos antiguos
rutas = ['data\\my_database.db']
funciones.eliminar_archivos(rutas)

# Conectar a SQLite y crear tablas
conn = sqlite3.connect('data\\my_database.db')
cur = conn.cursor()

# Insertar datos en tablas de SQLite
general_data.to_sql('general_data', conn, if_exists='replace', index=False)
employee_survey_data.to_sql('employee_survey_data', conn, if_exists='replace', index=False)
manager_survey_data.to_sql('manager_survey_data', conn, if_exists='replace', index=False)
retirement_info.to_sql('retirement_info', conn, if_exists='replace', index=False)

# Ejecutar script SQL para preprocesamiento
funciones.ejecutar_sql('Preprocesamiento.sql', cur)

# Cerrar la conexión
conn.close()
