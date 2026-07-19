import pyodbc
from app.core.config import CONNECTION_STRING

def get_db_connection():
    """Tạo kết nối tới SQL Server"""
    try:
        conn = pyodbc.connect(CONNECTION_STRING)
        return conn
    except Exception as e:
        print(f"❌ Lỗi kết nối SQL Server: {e}")
        return None