import sqlite3 as sql
import pandas as pd
import a_funciones as fn


con=sql.connect("data\\db_empleados.db")
cur=con.cursor()

df_employee=pd.read_csv("data\\db_employee.csv")



df_employee.to_sql('df_employee', con)

pd.read_sql("select* from df_employee", con)


cur.execute("select name from sqlite_master where type='table'")
cur.fetchall()

fn.ejecutar_sql('b_Preprocesamiento.sql')