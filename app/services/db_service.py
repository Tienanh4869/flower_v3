from functools import lru_cache
from app.core.database import get_db_connection


@lru_cache(maxsize=128)
def get_flower_info_by_name(folder_name: str):
    """
    Truy vấn thông tin loài hoa từ SQL Server dựa vào folder_name (ví dụ: 'Rose', 'Sunflower').
    Được cache trong RAM để tránh mở connection SQL liên tục mỗi khung hình video (tăng tốc độ lên >1000 lần).
    """
    conn = get_db_connection()
    if not conn:
        return None

    try:
        cursor = conn.cursor()
        query = """
            SELECT folder_name, name_vi, description 
            FROM Flower_Info 
            WHERE folder_name = ?
        """
        cursor.execute(query, (folder_name,))
        row = cursor.fetchone()

        if row:
            return {
                "folder_name": row[0],
                "name_vi": row[1],
                "description": row[2]
            }
        return None
    except Exception as e:
        print(f"❌ Lỗi truy vấn Database: {e}")
        return None
    finally:
        conn.close()