import os

# 1. Đường dẫn gốc của project (D:\Flower_v3)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2. CẤU HÌNH THƯ MỤC CHỨA MODEL VÀ DANH SÁCH 4 MÔ HÌNH YOLO 26
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
if not os.path.exists(WEIGHTS_DIR):
    WEIGHTS_DIR = os.path.join(BASE_DIR, "weight")

MODELS_CONFIG = {
    "yolo26n_detect": {
        "path": os.path.join(WEIGHTS_DIR, "yolo26n_detect.pt"),
        "task": "detect",
        "name": "YOLO 26 Nano (Detection)",
        "desc": "Nhận diện vị trí hoa, tốc độ xử lý nhanh gọn"
    },
    "yolo26s_detect": {
        "path": os.path.join(WEIGHTS_DIR, "yolo26s_detect.pt"),
        "task": "detect",
        "name": "YOLO 26 Small (Detection)",
        "desc": "Nhận diện vị trí hoa, độ chính xác cao"
    },
    "yolo26n_cls": {
        "path": os.path.join(WEIGHTS_DIR, "yolo26n_cls.pt"),
        "task": "cls",
        "name": "YOLO 26 Nano (Classification)",
        "desc": "Phân loại nhanh toàn bộ bức ảnh"
    },
    "yolo26s_cls": {
        "path": os.path.join(WEIGHTS_DIR, "yolo26s_cls.pt"),
        "task": "cls",
        "name": "YOLO 26 Small (Classification)",
        "desc": "Phân loại toàn bộ ảnh với độ chính xác cao"
    },
    "yolo26n_flower_only": {
        "path": os.path.join(WEIGHTS_DIR, "yolo26n_flower_only.pt") if os.path.exists(os.path.join(WEIGHTS_DIR, "yolo26n_flower_only.pt")) else os.path.join(WEIGHTS_DIR, "OnlyDecFlower.pt"),
        "task": "flower_only",
        "name": "YOLO 26 Nano (Flower-Only Detection)",
        "desc": "Quét và phát hiện tọa độ tất cả các bông hoa trong ảnh"
    }
}

MODEL_DETECT_PATH = MODELS_CONFIG["yolo26n_detect"]["path"]
print(f"[CONFIG CHECK] Thu muc weights: {WEIGHTS_DIR} | So model cau hinh: {len(MODELS_CONFIG)}")

# 3. SQL Server
SQL_SERVER_NAME = r"FREDDY\TIEN_ANH"
DATABASE_NAME = "Flower_in4"

CONNECTION_STRING = (
    f"Driver={{ODBC Driver 17 for SQL Server}};"
    f"Server={SQL_SERVER_NAME};"
    f"Database={DATABASE_NAME};"
    f"Trusted_Connection=yes;"
)