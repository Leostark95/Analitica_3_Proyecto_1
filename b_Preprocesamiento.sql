-- Crear tablas filtradas por año para empleados, general, manager, y retiros (2015 y 2016)
CREATE TABLE employee_2015 AS
SELECT * 
FROM employee
WHERE DateSurvey = '2015-12-31 00:00:00';

CREATE TABLE employee_2016 AS
SELECT * 
FROM employee
WHERE DateSurvey = '2016-12-31 00:00:00';

CREATE TABLE general_2015 AS 
SELECT * 
FROM general
WHERE InfoDate = '2015-12-31 00:00:00';

CREATE TABLE general_2016 AS 
SELECT * 
FROM general
WHERE InfoDate = '2016-12-31 00:00:00';

CREATE TABLE manager_2015 AS
SELECT * 
FROM manager
WHERE SurveyDate = '2015-12-31 00:00:00';

CREATE TABLE manager_2016 AS
SELECT * 
FROM manager
WHERE SurveyDate = '2016-12-31 00:00:00';

CREATE TABLE retirement_2015 AS
SELECT *
FROM retirement
WHERE strftime('%Y', retirementDate) = '2015';

CREATE TABLE retirement_2016 AS
SELECT *
FROM retirement
WHERE strftime('%Y', retirementDate) = '2016';

-- Unir tablas de 2015 y 2016
CREATE TABLE data_2015 AS
SELECT 
    g1.EmployeeID,
    g1.*, 
    e1.*, 
    m1.*, 
    CASE
        WHEN r2.retirementType = 'Resignation' THEN 1
        ELSE 0
    END AS renuncia2016
FROM 
    general_2015 g1
JOIN 
    employee_2015 e1 ON g1.EmployeeID = e1.EmployeeID
JOIN 
    manager_2015 m1 ON g1.EmployeeID = m1.EmployeeID
LEFT JOIN 
    retirement_2016 r2 ON g1.EmployeeID = r2.EmployeeID;

CREATE TABLE data_2016 AS
SELECT 
    g2.EmployeeID,
    g2.*, 
    e2.*, 
    m2.*
FROM 
    general_2016 g2
JOIN 
    employee_2016 e2 ON g2.EmployeeID = e2.EmployeeID
JOIN 
    manager_2016 m2 ON g2.EmployeeID = m2.EmployeeID
LEFT JOIN 
    retirement_2016 r2 ON g2.EmployeeID = r2.EmployeeID;

-- Eliminar registros de empleados que renunciaron en 2015
DELETE FROM data_2016
WHERE EXISTS (
    SELECT 1
    FROM retirement_2015 r1
    WHERE r1.EmployeeID = data_2016.EmployeeID
);


-- Renombrar la tabla data_2016 para predecir datos de 2017
ALTER TABLE data_2016 RENAME TO data_for_2017_prediction;

-- Crear tabla final consolidada para análisis
CREATE TABLE final_dataset AS
SELECT *
FROM data_2015
UNION ALL
SELECT *
FROM data_for_2017_prediction;