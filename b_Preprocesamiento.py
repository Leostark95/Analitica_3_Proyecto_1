import sqlite3
import pandas as pd
import os
import a_funciones as funciones # Este archivo contiene las funciones a utilizar
import sys

# --------------------------- Carga de datos --------------------------------------------------------
# Agregar la ruta que contiene el archivo de funciones
#sys.path
sys.path.append('C:/Users/delva/OneDrive - Universidad de Antioquia/SEMESTRE 2024-2/ANALITICA 3/Proyecto_Recursos_humanos/Analitica_3_Proyecto_RH')
#sys.path.append('') # Ruta Leo
#sys.path.append('c:\\Users\\Manuela\\Documents\\Analitica 3\\Analitica_3_Proyecto_1') # Ruta Manuela
#sys.path.append('') # Ruta Karen


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

#------------------ Vista previa de las bases de datos ------------------
columns = general_data.columns
columns
for i in columns:
    print('-'*10, i, '-'*10)
    print(general_data[i].value_counts())
'''
EmployeeCount y Over18 tienen el mismo dato en todos los registos,
 no aportan información útil para entrenar en el modelo predictivo
'''
#----------------- Eliminar variables que no son relevantes ---------------------
# Eliminar columnas sin información relevante
general_data = general_data.drop(
    [
        'EducationField', 'EmployeeCount', 'MaritalStatus', 'Over18', 
        'StandardHours', 'StockOptionLevel', 'YearsWithCurrManager', 
        'TotalWorkingYears'
    ], 
    axis=1
)

# Attrition tiene el mismo dato en todos los registros
retirement_info = retirement_info.drop(['Attrition', 'resignationReason'], axis = 1) 
retirement_info

#----------------- Verificar y eliminar nulos ---------------------

general_data.isnull().sum()
employee_survey_data.isnull().sum()
manager_survey_data.isnull().sum()
retirement_info.isnull().sum()

general_data = general_data.dropna()
employee_survey_data = employee_survey_data.dropna()

general_data.isnull().sum()
employee_survey_data.isnull().sum()


# Eliminar el archivo de la base de datos
if os.path.exists('my_database.db'):
    os.remove('my_database.db')
    os.remove('processed_data_2015.csv')
    os.remove('processed_data_2016.csv')
else:
    print("El archivo no existe")    

# Conectarse o crear la base de datos SQLite
conn = sqlite3.connect('my_database.db')

# Insertar los datos en tablas de SQLite
general_data.to_sql('general_data', conn, if_exists='replace', index=False)
employee_survey_data.to_sql('employee_survey_data', conn, if_exists='replace', index=False)
manager_survey_data.to_sql('manager_survey_data', conn, if_exists='replace', index=False)
retirement_info.to_sql('retirement_info', conn, if_exists='replace', index=False)

# Crear tablas filtradas
create_filtered_tables_sql = '''
CREATE TABLE IF NOT EXISTS employee_filtered AS
SELECT *
FROM employee_survey_data
WHERE DateSurvey IN ('2015-12-31', '2016-12-31');

CREATE TABLE IF NOT EXISTS general_filtered AS
SELECT *
FROM general_data
WHERE InfoDate IN ('2015-12-31', '2016-12-31');

CREATE TABLE IF NOT EXISTS manager_filtered AS
SELECT *
FROM manager_survey_data
WHERE SurveyDate IN ('2015-12-31', '2016-12-31');

CREATE TABLE IF NOT EXISTS retirement_filtered AS
SELECT *
FROM retirement_info
WHERE strftime('%Y', retirementDate) IN ('2015', '2016');
'''

# Ejecutar la creación de tablas filtradas
conn.executescript(create_filtered_tables_sql)

# Crear y unir datos de 2015
create_2015_sql = '''
CREATE TABLE IF NOT EXISTS tabla_2015 AS
SELECT g.EmployeeID, g.Age, g.BusinessTravel, g.Department, g.DistanceFromHome, g.Education, 
       g.JobRole, g.MonthlyIncome, g.NumCompaniesWorked, g.PercentSalaryHike, g.TrainingTimesLastYear, 
       g.YearsAtCompany, g.YearsSinceLastPromotion, e.EnvironmentSatisfaction, e.JobSatisfaction, 
       e.WorkLifeBalance, m.JobInvolvement, m.PerformanceRating, r.retirementType AS retiro_2016
FROM general_filtered g
JOIN employee_filtered e ON g.EmployeeID = e.EmployeeID AND e.DateSurvey = '2015-12-31'
JOIN manager_filtered m ON g.EmployeeID = m.EmployeeID AND m.SurveyDate = '2015-12-31'
LEFT JOIN retirement_filtered r ON g.EmployeeID = r.EmployeeID AND strftime('%Y', r.retirementDate) = '2016'
GROUP BY g.EmployeeID;
'''

# Crear y unir datos de 2016
create_2016_sql = '''
CREATE TABLE IF NOT EXISTS tabla_2016 AS
SELECT g.EmployeeID, g.Age, g.BusinessTravel, g.Department, g.DistanceFromHome, g.Education, 
       g.JobRole, g.MonthlyIncome, g.NumCompaniesWorked, g.PercentSalaryHike, g.TrainingTimesLastYear, 
       g.YearsAtCompany, g.YearsSinceLastPromotion, e.EnvironmentSatisfaction, e.JobSatisfaction, 
       e.WorkLifeBalance, m.JobInvolvement, m.PerformanceRating
FROM general_filtered g
JOIN employee_filtered e ON g.EmployeeID = e.EmployeeID AND e.DateSurvey = '2016-12-31'
JOIN manager_filtered m ON g.EmployeeID = m.EmployeeID AND m.SurveyDate = '2016-12-31'
GROUP BY g.EmployeeID;
'''

# Crear variable binaria para indicar renuncias en 2016
create_final_2015_sql = '''
CREATE TABLE IF NOT EXISTS tabla_final_2015 AS
SELECT *,
    CASE WHEN retiro_2016 = 'Resignation' THEN 1 ELSE 0 END AS renuncia2016
FROM tabla_2015;
'''


# Ejecutar el código SQL en partes
for sql_script in [
    create_2015_sql, create_2016_sql, create_final_2015_sql
]:
    try:
        conn.executescript(sql_script)
    except sqlite3.OperationalError as e:
        print(f"Error al ejecutar el script SQL: {e}")

# Consultar las tablas procesadas
result_2015 = pd.read_sql_query("SELECT * FROM tabla_final_2015;", conn)
result_2016 = pd.read_sql_query("SELECT * FROM tabla_2016;", conn)

# Guardar los resultados en archivos CSV
result_2015.to_csv('processed_data_2015.csv', index=False)
result_2016.to_csv('processed_data_2016.csv', index=False)

# Cerrar la conexión
conn.close()

# Imprimir la ubicación de los archivos generados
print("Los archivos se guardaron como:")
print("processed_data_2015.csv")
print("processed_data_2016.csv")