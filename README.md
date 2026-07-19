# 🌸 Flower v3 — 3-Mode Hybrid AI Studio

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![YOLO 26](https://img.shields.io/badge/YOLO%2026-Ultralytics-112233?style=for-the-badge)
![SQL Server](https://img.shields.io/badge/SQL%20Server-2019%2B-CC292B?style=for-the-badge&logo=microsoft-sql-server&logoColor=white)

**Flower v3 (3-Mode Hybrid AI Studio)** là hệ thống nhận diện, phân loại và tra cứu thông tin sinh học loài hoa thế hệ mới, được thiết kế theo chuẩn mực kiến trúc công nghiệp tách biệt hoàn toàn giữa **AI Core Engine / API Backend (FastAPI)** và **Giao diện Người dùng Hiện đại (Streamlit UI/UX)**.

---

## ✨ Các Tính Năng Nổi Bật (Key Features)

### 1. 🚀 3 Chế Độ Suy Luận Siêu Việt (3-Mode Inference Engine)
Hệ thống quản lý bộ 5 mô hình trọng lượng **YOLO 26 (`weights/`)** và hỗ trợ 3 chế độ hoạt động chuyên biệt:
- **`Mode 1` — Nhận diện 1 Pass (Single-Stage Detection)**: Sử dụng trực tiếp các mô hình `yolo26n_detect.pt` hoặc `yolo26s_detect.pt` để quét, khoanh vùng bounding box và định danh 15 loài hoa trong một lần suy luận duy nhất.
- **`Mode 2` — Phân loại toàn ảnh (Full-Image Classification)**: Sử dụng mô hình `yolo26n_cls.pt` hoặc `yolo26s_cls.pt` để phân loại nhanh toàn bộ bức ảnh đầu vào với độ chính xác và tốc độ tối đa.
- **`Mode 3` — Hai Giai Đoạn Lai (Two-Stage Hybrid Engine - Mới & Mạnh nhất)**:
  1. *Giai đoạn 1 (Detection)*: Chạy mô hình chuyên dụng `OnlyDecFlower.pt` (`yolo26n_flower_only`) chỉ làm nhiệm vụ phát hiện chính xác tọa độ $(x_1, y_1, x_2, y_2)$ của tất cả bông hoa trong ảnh.
  2. *Crop & Dynamic Padding*: Cắt từng bông hoa khỏi ảnh gốc kèm tỷ lệ mở rộng viền (*Crop Padding*) có thể điều chỉnh từ $0\%$ đến $30\%$ nhằm bảo toàn cấu trúc cánh hoa.
  3. *Giai đoạn 2 (Classification)*: Đưa từng ảnh crop vào mô hình `yolo26n_cls.pt` / `yolo26s_cls.pt` để định danh chuẩn xác tới từng bông hoa, khắc phục hoàn toàn hạn chế nhận diện nhầm của các mô hình detection đơn lẻ.

---

### 2. ⚡ Thanh Thống Kê Tốc Độ & Độ Trễ (Vietnamese Latency Breakdown Metrics)
Ngay dưới kết quả nhận diện trên tất cả các tab (Ảnh, Video và Webcam), hệ thống hiển thị thanh chỉ số hiệu năng được việt hóa hoàn toàn theo chuẩn mực công cụ đo kiểm của **Ultralytics Platform**:
```
⚡ Tiền xử lý: 1.5 ms  |  🤖 Suy luận AI: 18.2 ms  |  🎯 Hậu xử lý: 0.3 ms  |  ⏱️ Tổng thời gian: 20.0 ms (50.0 FPS)
```
Giúp người dùng, kỹ sư và khách hàng có cái nhìn trực quan, minh bạch nhất về tốc độ xử lý siêu nhạy của bộ đôi YOLO 26 + FastAPI.

---

### 3. 🔴 Live Stream Webcam Realtime (30 FPS, ~0s Delay) & Snapshot
- **Live Stream Realtime**: Kết nối trực tiếp camera Laptop/PC qua endpoint `/api/v1/detect_webcam_stream`, truyền tải luồng video MJPEG liên tục ở tốc độ ~30 FPS với độ trễ gần như bằng 0, đi kèm bảng đếm hoa theo thời gian thực (**📊 Live Flower Counter**) và chỉ số FPS trực tiếp.
- **Webcam Snapshot**: Chế độ chụp ảnh nhanh để phân tích sâu, hiển thị gallery từng bông hoa crop và thông số kỹ thuật.

---

### 4. 📚 Tích Hợp Cơ Sở Dữ Liệu SQL Server Tra Cứu Sinh Học
Mỗi kết quả nhận diện thành công đều được ánh xạ tự động tới SQL Server (`Flower_in4` / bảng `Flower_Info`) để truy xuất ngay lập tức:
- **Tên Tiếng Việt (`name_vi`)**: Hiển thị tên loài hoa bằng Tiếng Việt chuẩn mực.
- **Mô tả sinh học (`description`)**: Nguồn gốc, đặc tính sinh thái, ý nghĩa và hướng dẫn chăm sóc chi tiết.

---

### 5. 🎛️ Bảng Điều Khiển Nâng Cao (Navigator & Control Center)
- **📐 Kích thước ảnh đầu vào (`imgsz`)**: Điều chỉnh từ `32px` đến `1280px` để tối ưu cho hoa nhỏ ở xa hoặc đẩy cao FPS.
- **🔍 Diện tích Bounding Box tối thiểu (`min_box_area %`)**: Lọc bỏ các khung nhận diện nhiễu hoặc có kích thước quá nhỏ so với tổng diện tích ảnh.
- **🎯 Ngưỡng Tin Cậy (`Conf Threshold`) & Ngưỡng Gối Chồng (`IoU Threshold`)**.
- **✂️ Crop Padding**: Tùy chỉnh mở rộng khung viền khi crop hoa cho chế độ Hybrid.

---

## 📁 Cấu Trúc Dự Án (Project Structure)

```text
D:\Flower_v3\
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py           # Các API Endpoints (/detect, /detect_video, /detect_webcam_stream, ...)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py           # Cấu hình đường dẫn model, từ điển mô hình, chuỗi kết nối SQL Server
│   │   └── database.py         # Quản lý kết nối pyodbc tới SQL Server
│   ├── models/
│   │   └── __init__.py
│   └── services/
│       ├── __init__.py
│       ├── ai_service.py       # "Bộ não AI" — Quản lý suy luận YOLO 26, chế độ Hybrid, đo đạc Latency
│       └── db_service.py       # Truy xuất thông tin loài hoa từ SQL Server
├── frontend/
│   ├── __init__.py
│   └── app_ui.py               # Giao diện người dùng Streamlit UI/UX thế hệ mới
├── weights/
│   ├── yolo26n_detect.pt       # YOLO 26 Nano Detection
│   ├── yolo26s_detect.pt       # YOLO 26 Small Detection
│   ├── yolo26n_cls.pt          # YOLO 26 Nano Classification
│   ├── yolo26s_cls.pt          # YOLO 26 Small Classification
│   └── OnlyDecFlower.pt        # YOLO 26 Nano Flower-Only Detection (Dùng cho Giai đoạn 1 của Hybrid)
├── main.py                     # Entry point khởi chạy FastAPI Server (port 8000)
├── requirements.txt            # Danh sách thư viện Python cần thiết
├── .gitignore                  # Cấu hình bỏ qua file rác, virtual environment khi push Git
└── README.md                   # Tài liệu hướng dẫn sử dụng và triển khai
```

---

## 🛠️ Yêu Cầu & Cài Đặt Thư Viện (Installation)

### 1. Yêu cầu hệ thống
- **Hệ điều hành**: Windows 10/11 (64-bit)
- **Python**: 3.10 hoặc mới hơn
- **Driver SQL Server**: **Microsoft ODBC Driver 17 for SQL Server** (Tải và cài đặt miễn phí từ Microsoft nếu chưa có).

### 2. Tạo Virtual Environment & Cài đặt thư viện
Mở Terminal (PowerShell hoặc Command Prompt) tại thư mục `Flower_v3` và chạy lần lượt các lệnh sau:

```bash
# 1. Tạo môi trường ảo (virtual environment)
python -m venv .venv

# 2. Kích hoạt môi trường ảo (PowerShell trên Windows)
.\.venv\Scripts\activate

# 3. Nâng cấp pip và cài đặt toàn bộ phụ thuộc
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🗄️ Hướng Dẫn Cấu Hình Database SQL Server

### 1. Cài đặt Database và Bảng dữ liệu
Mở **SQL Server Management Studio (SSMS)**, kết nối vào server của bạn và chạy đoạn Script SQL dưới đây để tạo database `Flower_in4` và bảng `Flower_Info`:

```sql
CREATE DATABASE Flower_in4;
GO

USE Flower_in4;
GO

CREATE TABLE Flower_Info (
    id INT IDENTITY(1,1) PRIMARY KEY,
    folder_name NVARCHAR(100) NOT NULL UNIQUE,  -- Tên nhãn / folder trùng với output của YOLO
    name_vi NVARCHAR(200) NOT NULL,             -- Tên loài hoa bằng Tiếng Việt
    description NVARCHAR(MAX)                   -- Mô tả chi tiết loài hoa
);
GO

-- Thêm dữ liệu mẫu cho 15 loài hoa tiêu biểu
INSERT INTO Flower_Info (folder_name, name_vi, description)
VALUES 
('Rose', N'Hoa Hồng (Rose)', N'Hoa hồng là chúa tể các loài hoa, biểu tượng của tình yêu nồng cháy và vẻ đẹp kiêu sa. Thích hợp trồng nơi nhiều ánh sáng và thoát nước tốt.'),
('Sunflower', N'Hoa Hướng Dương (Sunflower)', N'Loài hoa luôn hướng về phía mặt trời, tượng trưng cho sự lạc quan, sức sống mãnh liệt và lòng trung thành kiên định.'),
('Tulip', N'Hoa Tulip (Uất Kim Hương)', N'Loài hoa nổi tiếng của Hà Lan với vẻ đẹp rực rỡ, sang trọng, mang thông điệp về một tình yêu hoàn hảo và sự thịnh vượng.'),
('Lily', N'Hoa Ly / Bách Hợp (Lily)', N'Hoa bách hợp mang hương thơm dịu ngào, thanh khiết, tượng trưng cho sự quý phái, đức hạnh và hạnh phúc đoàn viên.'),
('Daisy', N'Hoa Cúc Dại (Daisy)', N'Hoa cúc dại mang vẻ đẹp mộc mạc, tinh khôi, tượng trưng cho tình bạn chân thành, sự ngây thơ và kiên cường trước gió sương.'),
('Orchid', N'Hoa Phong Lan (Orchid)', N'Phong lan là vương giả của hoa cảnh với muôn vàn kiểu dáng độc đáo, tượng trưng cho vẻ đẹp quý phái, thanh lịch và sự hoàn mỹ.'),
('Dandelion', N'Hoa Bồ Công Anh (Dandelion)', N'Loài hoa dại với những cánh bay nhẹ theo gió, tượng trưng cho những ước mơ phiêu du, tự do và sự kiên cường vươn lên.'),
('Lotus', N'Hoa Sen (Lotus)', N'Quốc hoa của Việt Nam, biểu tượng cho sự thanh cao, thuần khiết, gần bùn mà chẳng hôi tanh mùi bùn.'),
('Marigold', N'Hoa Vạn Thọ (Marigold)', N'Loài hoa màu vàng cam rực rỡ thường xuất hiện trong các dịp lễ Tết, mang ý nghĩa cầu chúc sự trường thọ, bình an và may mắn.'),
('Carnation', N'Hoa Cẩm Chướng (Carnation)', N'Hoa cẩm chướng biểu trưng cho lòng biết ơn sâu sắc, tình yêu thương cao cả của người mẹ và sự kính trọng.'),
('Iris', N'Hoa Diên Vĩ (Iris)', N'Loài hoa mang tên vị thần cầu vồng Hy Lạp, biểu tượng của sự khôn ngoan, lòng dũng cảm và niềm hy vọng mãnh liệt.'),
('Lavender', N'Hoa Oải Hương (Lavender)', N'Loài hoa màu tím mộng mơ với hương thơm quyến rũ đặc trưng giúp thư giảm thần kinh, tượng trưng cho sự chung thủy.'),
('Jasmine', N'Hoa Nhài (Jasmine)', N'Hoa nhài trắng muốt tỏa hương thơm ngát về đêm, là biểu tượng của sự tinh khôi, êm dịu và tình cảm thuần khiết.'),
('Chrysanthemum', N'Hoa Cúc Đại Đóa (Chrysanthemum)', N'Loài hoa sang trọng mùa thu, tượng trưng cho sự trường thọ, niềm vui và sự cao quý trong văn hóa Đông Á.'),
('Hibiscus', N'Hoa Dâm Bụt (Hibiscus)', N'Loài hoa nhiệt đới màu sắc rực rỡ mang vẻ đẹp tươi trẻ, đầy sức sống và sự nhiệt huyết.');
GO
```

### 2. Cấu hình Chuỗi Kết Nối (`CONNECTION_STRING`)
Mở file `app/core/config.py`, kiểm tra và chỉnh sửa biến `SQL_SERVER_NAME` sao cho đúng với tên SQL Server trên máy bạn:
```python
# Ví dụ cấu hình trong app/core/config.py:
SQL_SERVER_NAME = r"FREDDY\TIEN_ANH"  # Thay bằng tên Server SQL của bạn (e.g., localhost, .\SQLEXPRESS)
DATABASE_NAME = "Flower_in4"
```

---

## ▶️ Hướng Dẫn Khởi Chạy Hệ Thống (Quick Start)

Để hệ thống hoạt động đầy đủ tính năng, bạn cần khởi chạy song song **2 Terminal** (1 cho Backend FastAPI và 1 cho Frontend Streamlit):

### 🌐 Bước 1: Khởi chạy Core Backend Engine (FastAPI)
Mở Terminal 1 (đã kích hoạt `.venv`) tại thư mục `Flower_v3` và chạy:
```bash
python main.py
```
*Lúc này, FastAPI Server sẽ online tại địa chỉ: `http://127.0.0.1:8000`.*
*(Bạn có thể truy cập `http://127.0.0.1:8000/docs` để xem tài liệu API Swagger UI tương tác trực tiếp).*

### 🖥️ Bước 2: Khởi chạy Giao diện Người dùng (Streamlit UI)
Mở Terminal 2 (đã kích hoạt `.venv`) tại thư mục `Flower_v3` và chạy:
```bash
streamlit run frontend/app_ui.py
```
*Trình duyệt sẽ tự động mở giao diện điều khiển Flower v3 tại địa chỉ: `http://localhost:8501`.*

---

## 💡 Hướng Dẫn Sử Dụng Nhanh trên Giao Diện

1. **Kiểm tra trạng thái**: Nhìn vào góc trên cùng của Sidebar bên trái, đảm bảo huy hiệu hiển thị `🟢 Backend FastAPI: Online`.
2. **Chọn Chế độ Suy luận**:
   - Nhấn chọn `Chế độ 3: Hai Giai Đoạn Lai (Hybrid Mode - Tốt nhất)` để đạt độ chính xác định danh tối đa cho từng bông hoa.
3. **Thử nghiệm với các Tab**:
   - **Tab 1 (`📷 Nhận diện từ Ảnh Upload`)**: Tải lên bức ảnh hoa bất kỳ từ máy tính -> Xem kết quả Bounding Box, Thư viện Gallery crop từng bông, chỉ số **⚡ Tiền xử lý | 🤖 Suy luận AI | 🎯 Hậu xử lý | ⏱️ Tổng thời gian** và thông tin SQL Server.
   - **Tab 2 (`🎥 Phân tích Video`)**: Tải lên đoạn video ngắn -> Xem luồng phát nhận diện realtime kèm thông số FPS tốc độ cao.
   - **Tab 3 (`📹 Camera Realtime`)**: Chọn **🔴 Live Stream Webcam Realtime** -> Bấm nút Bật và đưa hoa hoặc hình ảnh hoa ra trước camera Laptop để trải nghiệm tốc độ nhận diện 30 FPS không độ trễ!

---

## 📄 License & Author
Dự án được phát triển và tối ưu hóa phục vụ nghiên cứu & ứng dụng thực tế.
- **Repository**: [https://github.com/Tienanh4869/flower_v3.git](https://github.com/Tienanh4869/flower_v3.git)
