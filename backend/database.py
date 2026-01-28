
import mysql.connector
from mysql.connector import errorcode
from config import MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB, MYSQL_PORT

TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    age FLOAT,
    sex FLOAT,
    cp FLOAT,
    trestbps FLOAT,
    chol FLOAT,
    fbs FLOAT,
    restecg FLOAT,
    thalach FLOAT,
    exang FLOAT,
    oldpeak FLOAT,
    slope FLOAT,
    ca FLOAT,
    thal FLOAT,
    pred TINYINT,
    proba FLOAT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;
"""

def get_db():
    return mysql.connector.connect(
        host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASSWORD,
        database=MYSQL_DB, port=MYSQL_PORT
    )

def init_db():
    try:
        # Connect without specifying DB to ensure it exists
        cnx = mysql.connector.connect(host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASSWORD, port=MYSQL_PORT)
        cursor = cnx.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DB}")
        cursor.execute(f"USE {MYSQL_DB}")
        cursor.execute(TABLE_SQL)
        cnx.commit()
        cursor.close()
        cnx.close()
    except mysql.connector.Error as err:
        print(f"[DB INIT ERROR] {err}")

def insert_prediction(values_float, pred, proba):
    cols = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
    placeholders = ",".join(["%s"]* (len(cols)+2))  # + pred + proba
    sql = f"INSERT INTO predictions ({','.join(cols)}, pred, proba) VALUES ({placeholders})"
    cnx = get_db()
    cursor = cnx.cursor()
    cursor.execute(sql, (*values_float, pred, proba))
    cnx.commit()
    cursor.close()
    cnx.close()

def fetch_predictions(limit=100):
    cnx = get_db()
    cursor = cnx.cursor(dictionary=True)
    cursor.execute(f"SELECT * FROM predictions ORDER BY id DESC LIMIT %s", (limit,))
    rows = cursor.fetchall()
    cursor.close()
    cnx.close()
    return rows
