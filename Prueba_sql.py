import sqlite3
import pandas as pd
import os

# Eliminar el archivo de la base de datos
if os.path.exists('my_database.db'):
    os.remove('my_database.db')
else:
    print("El archivo no existe")



# Conectarse o crear la base de datos SQLite
conn = sqlite3.connect('my_database.db')

# Leer los archivos CSV con pandas
general_data = pd.read_csv('data/general_data.csv')
employee_survey_data = pd.read_csv('data/employee_survey_data.csv')
manager_survey_data = pd.read_csv('data/manager_survey.csv')
retirement_info = pd.read_csv('data/retirement_info.csv')

# Insertar los datos en tablas de SQLite
general_data.to_sql('general_data', conn, if_exists='replace', index=False)
employee_survey_data.to_sql('employee_survey_data', conn, if_exists='replace', index=False)
manager_survey_data.to_sql('manager_survey_data', conn, if_exists='replace', index=False)
retirement_info.to_sql('retirement_info', conn, if_exists='replace', index=False)

# Crear tablas filtradas

create_filtered_tables_sql ='''
CREATE TABLE IF NOT EXISTS employee_filtered AS
SELECT EmployeeID, DateSurvey, EnvironmentSatisfaction, JobSatisfaction, WorkLifeBalance
FROM employee_survey_data
WHERE DateSurvey IN ('2015-12-31', '2016-12-31');

CREATE TABLE IF NOT EXISTS general_filtered AS
SELECT EmployeeID, InfoDate, Age, BusinessTravel, Department, DistanceFromHome, Education, 
        JobRole, MonthlyIncome, NumCompaniesWorked, PercentSalaryHike, 
        TrainingTimesLastYear, YearsAtCompany, YearsSinceLastPromotion
FROM general_data
WHERE InfoDate IN ('2015-12-31', '2016-12-31');

CREATE TABLE IF NOT EXISTS manager_filtered AS
SELECT EmployeeID, SurveyDate, JobInvolvement, PerformanceRating
FROM manager_survey_data
WHERE SurveyDate IN ('2015-12-31', '2016-12-31');

CREATE TABLE IF NOT EXISTS retirement_filtered AS
SELECT EmployeeID, retirementDate, retirementType
FROM retirement_info
WHERE strftime('%Y', retirementDate) IN ('2015', '2016');
'''

# Ejecutar la creación de tablas filtradas
conn.executescript(create_filtered_tables_sql)