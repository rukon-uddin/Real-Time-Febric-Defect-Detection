import mysql.connector
import pandas as pd
import numpy as np

class Exec_Q:
    def __init__(self):
        self.mydb = mysql.connector.connect(
        host = "localhost",
        user = "root",
        passwd = "root",
        database = "defected_img")

        self.mycursor = self.mydb.cursor()

    
    def insert_row(self, INSERT_FORMULA, rows):
        self.mycursor.execute(INSERT_FORMULA, rows)
        self.mydb.commit()


    def delete_row(self, DELETE_FORMULA):
        self.mycursor.execute(DELETE_FORMULA)
        self.mydb.commit()

    def update_row(self, UPDATE_FORMULA):
        self.mycursor.execute(UPDATE_FORMULA)
        self.mydb.commit()

    def get_column(self, GET_FORMULA):
        self.mycursor.execute(GET_FORMULA)
        result = self.mycursor.fetchall()
        return result


    
