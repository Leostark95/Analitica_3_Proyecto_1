import sqlite3
import pandas as pd

# Conectarse o crear la base de datos SQLite
conn = sqlite3.connect('my_database.db')

# Leer los archivos CSV con pandas

general_data = pd.read_csv('data\\general_data.csv')
employee_survey_data = pd.read_csv('data\\employee_survey_data.csv')
manager_survey_data = pd.read_csv('data\\manager_survey.csv')
retirement_info = pd.read_csv('data\\retirement_info.csv')

# Insertar los datos en tablas de SQLite
general_data.to_sql('general_data', conn, if_exists='replace', index=False)
employee_survey_data.to_sql('employee_survey_data', conn, if_exists='replace', index=False)
manager_survey_data.to_sql('manager_survey_data', conn, if_exists='replace', index=False)
retirement_info.to_sql('retirement_info', conn, if_exists='replace', index=False)

# SQL query para crear las tablas y realizar las uniones
create_tables_sql = '''
-- Crear las tablas para empleados, general, manager y retiros por año (2015 y 2016)
WITH employee_filtered AS (
    SELECT EmployeeID, DateSurvey, EnvironmentSatisfaction, JobSatisfaction, WorkLifeBalance
    FROM employee_survey_data
    WHERE DateSurvey IN ('2015-12-31', '2016-12-31')
),
general_filtered AS (
    SELECT EmployeeID, InfoDate, Age, BusinessTravel, Department
    FROM general_data
    WHERE InfoDate IN ('2015-12-31', '2016-12-31')
),
manager_filtered AS (
    SELECT EmployeeID, SurveyDate, JobInvolvement, PerformanceRating
    FROM manager_survey_data
    WHERE SurveyDate IN ('2015-12-31', '2016-12-31')
),
retirement_filtered AS (
    SELECT EmployeeID, retirementDate, retirementType
    FROM retirement_info
    WHERE strftime('%Y', retirementDate) IN ('2015', '2016')
)

-- Unimos los datos de 2015 en una tabla
CREATE TABLE tabla_2015 AS
SELECT g.EmployeeID, g.Age, g.BusinessTravel, g.Department, e.EnvironmentSatisfaction, e.JobSatisfaction, 
       e.WorkLifeBalance, m.JobInvolvement, m.PerformanceRating, r.retirementType AS retiro_2015
FROM general_filtered g
JOIN employee_filtered e ON g.EmployeeID = e.EmployeeID AND e.DateSurvey = '2015-12-31'
JOIN manager_filtered m ON g.EmployeeID = m.EmployeeID AND m.SurveyDate = '2015-12-31'
LEFT JOIN retirement_filtered r ON g.EmployeeID = r.EmployeeID AND strftime('%Y', r.retirementDate) = '2015';

-- Unimos los datos de 2016 en una tabla
CREATE TABLE tabla_2016 AS
SELECT g.EmployeeID, g.Age, g.BusinessTravel, g.Department, e.EnvironmentSatisfaction, e.JobSatisfaction, 
       e.WorkLifeBalance, m.JobInvolvement, m.PerformanceRating, r.retirementType AS retiro_2016
FROM general_filtered g
JOIN employee_filtered e ON g.EmployeeID = e.EmployeeID AND e.DateSurvey = '2016-12-31'
JOIN manager_filtered m ON g.EmployeeID = m.EmployeeID AND m.SurveyDate = '2016-12-31'
LEFT JOIN retirement_filtered r ON g.EmployeeID = r.EmployeeID AND strftime('%Y', r.retirementDate) = '2016';

-- Crear la variable objetivo para 2015
CREATE TABLE retiros_2016 AS
SELECT EmployeeID, retiro_2016
FROM tabla_2016;

-- Crear tabla para unir datos de 2015 con la variable objetivo
CREATE TABLE tabla_20150 AS
SELECT t.*, r.retiro_2016
FROM tabla_2015 t
LEFT JOIN retiros_2016 r ON t.EmployeeID = r.EmployeeID;

-- Crear variable binaria en la tabla de 2015 para indicar renuncias en 2016
CREATE TABLE tabla_final_2015 AS
SELECT *,
    CASE WHEN retiro_2016 = 'Resignation' THEN 1 ELSE 0 END AS renuncia2016
FROM tabla_20150;

-- Limpiar columnas innecesarias en la tabla 2015
CREATE TABLE tabla_limpia_2015 AS
SELECT EmployeeID, Age, BusinessTravel, Department, EnvironmentSatisfaction, JobSatisfaction, 
       WorkLifeBalance, JobInvolvement, PerformanceRating, renuncia2016
FROM tabla_final_2015;

-- Eliminar empleados que renunciaron en 2015 de los datos de 2016
DELETE FROM tabla_2016
WHERE EmployeeID IN (SELECT EmployeeID FROM tabla_2015 WHERE retiro_2015 = 'Resignation');

-- Limpiar columnas innecesarias en la tabla 2016
CREATE TABLE tabla_limpia_2016 AS
SELECT EmployeeID, Age, BusinessTravel, Department, EnvironmentSatisfaction, JobSatisfaction, 
       WorkLifeBalance, JobInvolvement, PerformanceRating
FROM tabla_2016;

-- Renombrar tabla final 2016 para predecir
ALTER TABLE tabla_limpia_2016 RENAME TO tabla_prediccion_2016;
'''

# Ejecutar el código SQL
conn.executescript(create_tables_sql)

# Consultar las tablas procesadas
result_2015 = pd.read_sql_query("SELECT * FROM tabla_limpia_2015;", conn)
result_2016 = pd.read_sql_query("SELECT * FROM tabla_prediccion_2016;", conn)

# Guardar los resultados en archivos CSV
result_2015.to_csv('processed_employee_data_2015.csv', index=False)
result_2016.to_csv('processed_employee_data_2016.csv', index=False)

# Cerrar la conexión
conn.close()

# Imprimir la ubicación de los archivos generados
print("Los archivos se guardaron como:")
print("processed_employee_data_2015.csv")
print("processed_employee_data_2016.csv")