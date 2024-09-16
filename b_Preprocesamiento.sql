-- Crear las tablas para empleados, general, manager y retiros por año (2015 y 2016)
-- Eliminar tablas si ya existen
DROP TABLE IF EXISTS employee_filtered;
DROP TABLE IF EXISTS general_filtered;
DROP TABLE IF EXISTS manager_filtered;
DROP TABLE IF EXISTS retirement_filtered;
DROP TABLE IF EXISTS tabla_2015;
DROP TABLE IF EXISTS tabla_2016;
DROP TABLE IF EXISTS tabla_final_2015;

-- Crear tablas usando CTE
CREATE TABLE IF NOT EXISTS employee_filtered AS
WITH employee_filtered_cte AS (
    SELECT *
    FROM employee_survey_data
    WHERE DateSurvey IN ('2015-12-31', '2016-12-31')
)
SELECT * FROM employee_filtered_cte;

CREATE TABLE IF NOT EXISTS general_filtered AS
WITH general_filtered_cte AS (
    SELECT *
    FROM general_data
    WHERE InfoDate IN ('2015-12-31', '2016-12-31')
)
SELECT * FROM general_filtered_cte;

CREATE TABLE IF NOT EXISTS manager_filtered AS
WITH manager_filtered_cte AS (
    SELECT *
    FROM manager_survey_data
    WHERE SurveyDate IN ('2015-12-31', '2016-12-31')
)
SELECT * FROM manager_filtered_cte;

CREATE TABLE IF NOT EXISTS retirement_filtered AS
WITH retirement_filtered_cte AS (
    SELECT *
    FROM retirement_info
    WHERE strftime('%Y', retirementDate) IN ('2015', '2016')
)
SELECT * FROM retirement_filtered_cte;

-- Crear tabla de 2015
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

-- Crear tabla de 2016
CREATE TABLE IF NOT EXISTS processed_data_2016 AS
SELECT g.EmployeeID, g.Age, g.BusinessTravel, g.Department, g.DistanceFromHome, g.Education, 
       g.JobRole, g.MonthlyIncome, g.NumCompaniesWorked, g.PercentSalaryHike, g.TrainingTimesLastYear, 
       g.YearsAtCompany, g.YearsSinceLastPromotion, e.EnvironmentSatisfaction, e.JobSatisfaction, 
       e.WorkLifeBalance, m.JobInvolvement, m.PerformanceRating
FROM general_filtered g
JOIN employee_filtered e ON g.EmployeeID = e.EmployeeID AND e.DateSurvey = '2016-12-31'
JOIN manager_filtered m ON g.EmployeeID = m.EmployeeID AND m.SurveyDate = '2016-12-31'
GROUP BY g.EmployeeID;

-- Crear tabla final de 2015 con variable binaria
CREATE TABLE IF NOT EXISTS processed_data_2015 AS
SELECT *,
    CASE WHEN retiro_2016 = 'Resignation' THEN 1 ELSE 0 END AS renuncia2016
FROM tabla_2015;