import streamlit as st
import requests
import base64
from PIL import Image
import io
import json
from collections import Counter

# 1. Cấu hình trang Streamlit
st.set_page_config(
    page_title="YOLO 26",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Nhúng Custom Vanilla CSS tạo phong cách Navigator Bar hiện đại & Clean Card Layout
st.markdown("""
<style>
/* Custom Top Navigation Bar (Navigator Bar / Tabs) */
div[data-baseweb="tab-list"] {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 16px;
    padding: 8px 12px;
    gap: 12px;
    margin-bottom: 1.8rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
}
button[data-baseweb="tab"] {
    border-radius: 12px !important;
    padding: 12px 26px !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #cccccc !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.25s ease !important;
}
button[data-baseweb="tab"]:hover {
    background: rgba(255, 255, 255, 0.08) !important;
    color: #ffffff !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #ff758c 0%, #8e44ad 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 15px rgba(255, 117, 140, 0.4) !important;
}
div[data-baseweb="tab-border"], div[data-baseweb="tab-highlight"] {
    display: none !important;
}

/* Thẻ thông tin loài hoa */
.flower-card {
    background: rgba(255, 255, 255, 0.05);
    border-left: 5px solid #ff758c;
    padding: 1.2rem 1.5rem;
    border-radius: 10px;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

/* Status Pill */
.status-pill {
    display: inline-block;
    padding: 0.4rem 1rem;
    border-radius: 50px;
    font-size: 0.88rem;
    font-weight: 600;
    margin-top: 0.3rem;
}
.status-online {
    background-color: #2ecc71;
    color: white;
}
.status-offline {
    background-color: #e74c3c;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Địa chỉ API Backend (FastAPI)
API_BASE_URL = "http://127.0.0.1:8000/api/v1"
DETECT_URL = f"{API_BASE_URL}/detect"
DETECT_VIDEO_URL = f"{API_BASE_URL}/detect_video"
DETECT_VIDEO_STREAM_URL = f"{API_BASE_URL}/detect_video_stream"
DETECT_WEBCAM_STREAM_URL = f"{API_BASE_URL}/detect_webcam_stream"
MODELS_URL = f"{API_BASE_URL}/models"

# 3. Sidebar - Bảng điều khiển YOLO 26 được cải tiến (Navigator & Control Center)
with st.sidebar:
    st.markdown("##BẢNG ĐIỀU KHIỂN")
    st.markdown("---")

    # Kiểm tra trạng thái Backend
    backend_ok = False
    try:
        res_models = requests.get(MODELS_URL, timeout=2)
        if res_models.status_code == 200:
            backend_ok = True
    except Exception:
        backend_ok = False

    if backend_ok:
        st.markdown('<div class="status-pill status-online">🟢 Backend FastAPI: Online</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-pill status-offline">🔴 Backend FastAPI: Offline</div>', unsafe_allow_html=True)
        st.error("⚠️ Không thể kết nối tới Backend. Hãy chạy `python main.py` trước!")

    st.markdown("### Chế độ Suy luận")
    task_choice = st.radio(
        "Chọn bài toán:",
        options=[
            "Detection - 15 loài",
            "Classification - 15 loài",
            "Hai giai đoạn Lai (Detection ➔ Crop ➔ Classify)"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Cấu hình Mô hình & Ngưỡng")

    if "Hai giai đoạn" in task_choice or "Crop" in task_choice or "Hybrid" in task_choice:
        # Chế độ Hai giai đoạn Lai (Hybrid)
        task_mode = "hybrid"
        model_key = "yolo26n_flower_only"
        st.markdown("👉 **Stage 1 (Detection)**: Mặc định `yolo26n_flower_only` *(chuyên nhận diện hoa)*")
        cls_size_choice = st.selectbox(
            "Chọn Model Classification (Stage 2):",
            options=["yolo26s_cls", "yolo26n_cls"],
            index=0
        )
        cls_model_key = cls_size_choice
        st.info(f"👉 **Pipeline**: `{model_key}` ➔ Crop ➔ `{cls_model_key}`")

    elif "Classification" in task_choice:
        task_mode = "cls"
        cls_model_key = "yolo26s_cls"
        size_choice = st.selectbox(
            "Chọn Model Classification:",
            options=["yolo26n_cls", "yolo26s_cls"],
            index=0
        )
        model_key = size_choice
        st.info(f"👉 **Model Classification**: `{model_key}.pt`")

    else:
        # Chế độ Nhận diện 1 Pass (Detection)
        task_mode = "detect"
        cls_model_key = "yolo26s_cls"
        size_choice = st.selectbox(
            "Chọn Model Detection:",
            options=["yolo26n", "yolo26s"],
            index=0
        )
        model_key = f"{size_choice}_detect"
        st.info(f"👉 **Model Detection**: `{model_key}.pt`")

    st.markdown("#### 🎛️ Tham số Kỹ thuật")
    
    # Crop padding chỉ áp dụng ở chế độ Hybrid
    is_not_hybrid = (task_mode != "hybrid")
    crop_padding_pct = st.slider("Độ mở rộng viền cắt (Crop Padding %):", 0, 30, 10, 1, disabled=is_not_hybrid)
    crop_padding = crop_padding_pct / 100.0 if not is_not_hybrid else 0.05
    if is_not_hybrid:
        st.caption("🔒 *Crop Padding bị khóa vì chỉ áp dụng cho chế độ Hai giai đoạn Lai.*")

    # Bounding box params (conf, iou, min_area) không áp dụng ở chế độ Classification
    is_cls = (task_mode == "cls")
    conf_threshold = st.slider("Độ tin cậy tối thiểu (Conf Threshold):", 0.10, 0.95, 0.40, 0.05, disabled=is_cls)
    iou_threshold = st.slider("Độ chồng lấp Bounding Box (IoU Threshold):", 0.10, 0.90, 0.45, 0.05, disabled=is_cls)
    min_box_area_pct = st.slider("Diện tích Bounding Box tối thiểu (%):", 0.0, 10.0, 0.0, 0.1, disabled=is_cls)
    min_box_area = min_box_area_pct if not is_cls else 0.0
    if is_cls:
        st.caption("🔒 *Các tham số Bounding Box (Conf, IoU, Min Area) bị khóa ở chế độ Phân loại toàn ảnh.*")

    imgsz = st.slider("Kích thước đầu vào (Image Size - imgsz):", min_value=32, max_value=1280, value=640, step=32)

    st.markdown("---")
    st.caption("Flower — YOLO 26 + SQL Server")

# 5. Khu vực Navigation Bar chính chia thành 3 Tab Pill hiện đại
tab_image, tab_video, tab_webcam = st.tabs([
    "📸 Demo với Ảnh (Image Upload)",
    "🎥 Demo với Video (Video Upload)",
    "📹 Camera Realtime (Webcam Input)"
])

# ==============================================================================
# TAB 1: DEMO VỚI ẢNH TẢI LÊN
# ==============================================================================
with tab_image:
    st.markdown("### 📁 Tải lên bức ảnh hoa cần phân tích")
    uploaded_file = st.file_uploader("Chọn ảnh định dạng JPG, PNG, WEBP...", type=["jpg", "jpeg", "png", "webp"], key="uploader_img")

    if uploaded_file is not None:
        col_in, col_out = st.columns([1, 1])
        
        with col_in:
            st.markdown("#### 🖼️ Ảnh gốc tải lên")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

            analyze_btn = st.button("🚀 PHÂN TÍCH ẢNH NGAY", type="primary", use_container_width=True, key="btn_img")

        with col_out:
            st.markdown("#### 🎯 Kết quả suy luận YOLO 26")
            if analyze_btn:
                with st.spinner("🤖 YOLO 26 đang phân tích và tra cứu SQL Server..."):
                    try:
                        uploaded_file.seek(0)
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        data = {
                            "model_key": model_key,
                            "conf_threshold": str(conf_threshold),
                            "iou_threshold": str(iou_threshold),
                            "task_mode": task_mode,
                            "cls_model_key": cls_model_key,
                            "crop_padding": str(crop_padding),
                            "imgsz": str(imgsz),
                            "min_box_area": str(min_box_area)
                        }
                        response = requests.post(DETECT_URL, files=files, data=data)

                        if response.status_code == 200:
                            res_json = response.json()
                            st.session_state["img_result"] = res_json
                            st.session_state["last_img_name"] = uploaded_file.name
                        else:
                            st.error(f"❌ Lỗi từ Server: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Không thể kết nối Backend: {e}")

            if "img_result" in st.session_state and st.session_state.get("last_img_name") == uploaded_file.name:
                res = st.session_state["img_result"]
                if "image_base64" in res:
                    img_data = res["image_base64"].split(",")[1]
                    img_bytes = base64.b64decode(img_data)
                    plotted_img = Image.open(io.BytesIO(img_bytes))
                    st.image(plotted_img, use_container_width=True)

                    speed = res.get("speed_metrics", {})
                    if speed:
                        st.markdown(f"""
                        <div style="background: #1e1e24; padding: 10px 14px; border-radius: 10px; margin-top: 10px; border: 1px solid #333; display: flex; justify-content: space-around; text-align: center; color: white;">
                            <div><span style="font-size: 12px; color: #aaa;">⚡ Tiền xử lý</span><br><b>{speed.get('preprocess', 0)} ms</b></div>
                            <div><span style="font-size: 12px; color: #aaa;">🤖 Suy luận AI</span><br><b>{speed.get('inference', 0)} ms</b></div>
                            <div><span style="font-size: 12px; color: #aaa;">🎯 Hậu xử lý</span><br><b>{speed.get('postprocess', 0)} ms</b></div>
                            <div><span style="font-size: 12px; color: #aaa;">⏱️ Tổng thời gian</span><br><b style="color: #00ff88;">{speed.get('total', 0)} ms ({speed.get('fps', 0)} FPS)</b></div>
                        </div>
                        """, unsafe_allow_html=True)

        # Hiển thị thông tin SQL Server phía dưới kèm Độ chính xác Detection và Classification
        if "img_result" in st.session_state and st.session_state.get("last_img_name") == uploaded_file.name:
            res = st.session_state["img_result"]
            st.markdown("---")
            st.markdown("### 📚 Tra Cứu Thông Tin & Thư Viện Chi Tiết (SQL Server)")

            if res["task"] in ["detect", "hybrid"]:
                items = res.get("detected_items", [])
                if len(items) > 0:
                    st.success(f"🎉 Phát hiện được **{len(items)}** bông hoa/đối tượng trong ảnh!")
                    
                    st.markdown("#### 🌺 Thư viện Chi tiết Từng Bông Hoa (Cropped Gallery)")
                    cols = st.columns(3)
                    for idx, item in enumerate(items):
                        db_info = item.get("db_info", {})
                        name_vi = db_info.get("name_vi", item["folder_name"])
                        desc = db_info.get("description", "Chưa có thông tin mô tả chi tiết.")
                        conf_cls_pct = item["confidence"] * 100
                        conf_det_pct = item.get("conf_det", item["confidence"]) * 100

                        with cols[idx % 3]:
                            st.markdown(f'<div class="flower-card">', unsafe_allow_html=True)
                            st.markdown(f"#### 🌺 #{idx + 1}: {name_vi}")
                            
                            # Hiển thị độ chính xác Phát hiện (Detection Conf) và Phân loại (Classification Conf)
                            if res["task"] == "hybrid" and item.get("conf_det") is not None:
                                st.markdown(f"🎯 **Phát hiện (Det)**: `{conf_det_pct:.1f}%`  |  🏷️ **Phân loại (Cls)**: `{conf_cls_pct:.1f}%`")
                            else:
                                st.markdown(f"🎯 **Độ chính xác Detection**: `{conf_det_pct:.1f}%`")

                            if item.get("crop_base64"):
                                crop_bytes = base64.b64decode(item["crop_base64"].split(",")[1])
                                st.image(Image.open(io.BytesIO(crop_bytes)), use_container_width=True)
                            
                            st.markdown(f"**Tên khoa học / Folder**: `{item['folder_name']}`")
                            
                            if item.get("top3") and len(item["top3"]) > 0:
                                st.markdown("📊 **Top 3 Phân loại:**")
                                for t in item["top3"]:
                                    t_name = t.get("name_vi", t["folder_name"])
                                    t_conf = t["confidence"]
                                    st.write(f"- **{t_name}**: `{t_conf*100:.1f}%`")
                                    st.progress(t_conf)

                            with st.expander("📖 Xem mô tả chi tiết", expanded=False):
                                st.markdown(f"> {desc}")
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Model AI không tìm thấy loài hoa nào trong ảnh với độ tin cậy được chọn.")
            else:
                top1 = res.get("top1")
                if top1:
                    db_info = top1.get("db_info", {})
                    name_vi = db_info.get("name_vi", top1["folder_name"])
                    desc = db_info.get("description", "Chưa có mô tả chi tiết.")
                    conf_pct = top1["confidence"] * 100

                    st.success(f"🎉 Phân loại chính: **{name_vi}** (Độ chính xác: **{conf_pct:.2f}%**)")
                    
                    col_t1, col_t2 = st.columns([2, 1])
                    with col_t1:
                        st.markdown(f"#### 📖 Thông tin sinh học & ý nghĩa:")
                        st.markdown(f"> {desc}")
                    with col_t2:
                        st.markdown("#### 📊 Top 5 khả năng cao nhất:")
                        for item in res.get("top5", []):
                            t_name = item["db_info"].get("name_vi", item["folder_name"]) if item.get("db_info") else item["folder_name"]
                            t_conf = item["confidence"]
                            st.write(f"**{t_name}** (`{t_conf*100:.1f}%`)")
                            st.progress(t_conf)

# ==============================================================================
# TAB 2: DEMO VỚI VIDEO TẢI LÊN
# ==============================================================================
with tab_video:
    st.markdown("### 🎬 Tải lên đoạn Video hoa cần xử lý (Live Streaming)")
    uploaded_video = st.file_uploader("Chọn video định dạng MP4, MOV, AVI...", type=["mp4", "mov", "avi"], key="uploader_vid")

    if uploaded_video is not None:
        st.markdown("#### ⚙️ Chế độ Phát Trực Tiếp (MJPEG Live Streaming)")
        skip_choice = st.selectbox(
            "Tối ưu tốc độ (Nhảy khung hình - Frame Skip):",
            options=[
                "1 (Xử lý toàn bộ frame - Chuẩn)",
                "2 (Nhảy cóc 1 frame - Tăng tốc 2x)",
                "3 (Nhảy cóc 2 frame - Tăng tốc siêu nhanh 3x)"
            ],
            index=1
        )
        skip_val = int(skip_choice.split(" ")[0])
        
        start_stream = st.button("⚡ PHÂN TÍCH VIDEO NGAY (LIVE STREAMING - 0s DELAY)", type="primary", use_container_width=True, key="btn_vid_stream")
        
        st.markdown("---")
        col_vid_in, col_vid_out = st.columns(2)
        
        with col_vid_in:
            st.markdown("#### 📽️ Video Gốc (Input)")
            st.video(uploaded_video)
            
        with col_vid_out:
            live_header = st.empty()
            live_counter_placeholder = st.empty()
            live_placeholder = st.empty()
            live_header.markdown("#### 🤖 Video AI Nhận Diện (Live Engine)")
            if not start_stream:
                live_placeholder.info("👈 Bấm nút **PHÂN TÍCH VIDEO NGAY** phía trên để xem kết quả AI phát trực tiếp tại cột này!")

        stats_placeholder = st.empty()
        
        if start_stream:
            live_header.markdown("#### 🔴 Đang phát trực tiếp kết quả YOLO 26:")
            try:
                uploaded_video.seek(0)
                files = {"file": (uploaded_video.name, uploaded_video.getvalue(), uploaded_video.type)}
                data = {
                    "model_key": model_key,
                    "conf_threshold": str(conf_threshold),
                    "iou_threshold": str(iou_threshold),
                    "skip_frames": str(skip_val),
                    "task_mode": task_mode,
                    "cls_model_key": cls_model_key,
                    "crop_padding": str(crop_padding)
                }
                
                response = requests.post(DETECT_VIDEO_STREAM_URL, files=files, data=data, stream=True)

                if response.status_code == 200:
                    bytes_buffer = bytes()
                    for chunk in response.iter_content(chunk_size=8192):
                        bytes_buffer += chunk
                        while True:
                            a = bytes_buffer.find(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n')
                            j = bytes_buffer.find(b'--frame\r\nContent-Type: application/json\r\n\r\n')

                            if a != -1 and (j == -1 or a < j):
                                b = bytes_buffer.find(b'\r\n--frame', a + 37)
                                if b != -1:
                                    jpg_data = bytes_buffer[a + 37:b]
                                    bytes_buffer = bytes_buffer[b:]
                                    live_placeholder.image(jpg_data, use_container_width=True)
                                else:
                                    break
                            elif j != -1:
                                b = bytes_buffer.find(b'\r\n--frame', j + 43)
                                if b != -1 or len(bytes_buffer) > j + 43 + 10:
                                    json_bytes = bytes_buffer[j + 43:b] if b != -1 else bytes_buffer[j + 43:].strip()
                                    try:
                                        vid_res = json.loads(json_bytes.decode("utf-8"))
                                        if "error" in vid_res:
                                            stats_placeholder.error(f"❌ Lỗi từ Server: {vid_res['error']}")
                                        elif "live_counts" in vid_res:
                                            counts_dict = vid_res["live_counts"]
                                            if len(counts_dict) > 0:
                                                pills = " | ".join([f"**{k}**: `{v}`" for k, v in counts_dict.items()])
                                                live_counter_placeholder.markdown(
                                                    f'<div style="padding: 0.6rem 1rem; background: rgba(255, 117, 140, 0.15); border-left: 4px solid #ff758c; border-radius: 8px; margin-bottom: 0.8rem;">📊 **Live Flower Counter**: {pills}</div>',
                                                    unsafe_allow_html=True
                                                )
                                        elif vid_res.get("status") == "success":
                                            live_header.markdown(f"#### ✅ Đã phát trực tiếp xong (**{vid_res.get('total_frames', 0)} frames**)")
                                            with stats_placeholder.container():
                                                st.markdown("---")
                                                st.markdown("#### 📈 Thống kê tần suất xuất hiện loài hoa trong Video:")
                                                summary_list = vid_res.get("summary", [])
                                                if len(summary_list) > 0:
                                                    for row in summary_list:
                                                        st.write(f"- **{row['name_vi']}** (Folder: `{row['folder_name']}`): Xuất hiện trong **{row['occurrences']} frames**")
                                                else:
                                                    st.warning("⚠️ Không tìm thấy đối tượng hoa nào trong video.")
                                    except Exception as ex:
                                        pass
                                    bytes_buffer = bytes()
                                    break
                                else:
                                    break
                            else:
                                break
                else:
                    st.error(f"❌ Lỗi từ Server: {response.text}")
            except Exception as e:
                st.error(f"❌ Lỗi kết nối tới Backend khi stream video: {e}")

# ==============================================================================
# ==============================================================================
# TAB 3: CAMERA REALTIME (WEBCAM INPUT)
# ==============================================================================
with tab_webcam:
    st.markdown("### 📹 Nhận diện Hoa trực tiếp qua Webcam Realtime")
    webcam_mode = st.radio(
        "Chọn chế độ hoạt động của Camera:",
        ["🔴 Live Stream Webcam Realtime (30 FPS - Liên tục bám sát hoa)", "📸 Chụp từng ảnh (Webcam Snapshot)"],
        horizontal=True,
        key="radio_webcam_mode"
    )

    if "🔴 Live Stream" in webcam_mode:
        st.markdown("💡 *Chế độ Live Stream sẽ mở trực tiếp camera PC/Laptop qua FastAPI và phát liên tục 30 FPS với độ trễ ~0s! Bạn chỉ cần giơ bông hoa ra trước ống kính.*")
        col_cam_ctrl1, col_cam_ctrl2 = st.columns([1, 2])
        with col_cam_ctrl1:
            cam_idx = st.number_input("Chỉ số Camera (Camera Index):", min_value=0, max_value=5, value=0, step=1, help="0 là camera mặc định của Laptop/PC")
            btn_start_webcam = st.button("🔴 BẬT LIVE STREAM WEBCAM", type="primary", use_container_width=True, key="btn_live_webcam")
        with col_cam_ctrl2:
            live_cam_status = st.empty()
            live_cam_counter = st.empty()
            live_cam_speed = st.empty()

        live_cam_frame = st.empty()

        if btn_start_webcam:
            live_cam_status.markdown("#### 🔴 Đang phát Live Stream từ Webcam...")
            try:
                params = {
                    "model_key": model_key,
                    "conf_threshold": str(conf_threshold),
                    "iou_threshold": str(iou_threshold),
                    "skip_frames": "1",
                    "task_mode": task_mode,
                    "cls_model_key": cls_model_key,
                    "crop_padding": str(crop_padding),
                    "imgsz": str(imgsz),
                    "min_box_area": str(min_box_area),
                    "camera_index": str(cam_idx)
                }
                resp_cam = requests.get(DETECT_WEBCAM_STREAM_URL, params=params, stream=True, timeout=10)
                if resp_cam.status_code == 200:
                    bytes_buf = bytes()
                    for chunk in resp_cam.iter_content(chunk_size=8192):
                        bytes_buf += chunk
                        while True:
                            a = bytes_buf.find(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n')
                            j = bytes_buf.find(b'--frame\r\nContent-Type: application/json\r\n\r\n')

                            if a != -1 and (j == -1 or a < j):
                                b = bytes_buf.find(b'\r\n--frame', a + 37)
                                if b != -1:
                                    jpg_data = bytes_buf[a + 37:b]
                                    bytes_buf = bytes_buf[b:]
                                    live_cam_frame.image(jpg_data, use_container_width=True)
                                else:
                                    break
                            elif j != -1:
                                b = bytes_buf.find(b'\r\n--frame', j + 43)
                                if b != -1 or len(bytes_buf) > j + 43 + 10:
                                    json_bytes = bytes_buf[j + 43:b] if b != -1 else bytes_buf[j + 43:].strip()
                                    try:
                                        c_info = json.loads(json_bytes.decode("utf-8"))
                                        if "error" in c_info:
                                            live_cam_status.error(f"❌ Lỗi từ Server: {c_info['error']}")
                                        elif "live_counts" in c_info:
                                            cnt_dict = c_info["live_counts"]
                                            if len(cnt_dict) > 0:
                                                pills = " | ".join([f"**{k}**: `{v}`" for k, v in cnt_dict.items()])
                                                live_cam_counter.markdown(
                                                    f'<div style="padding: 0.6rem 1rem; background: rgba(46, 204, 113, 0.15); border-left: 4px solid #2ecc71; border-radius: 8px;">📊 **Live Flower Counter**: {pills}</div>',
                                                    unsafe_allow_html=True
                                                )
                                            spd = c_info.get("speed_metrics", {})
                                            if spd:
                                                live_cam_speed.markdown(f"""
                                                <div style="background: #1e1e24; padding: 8px 12px; border-radius: 8px; border: 1px solid #333; display: flex; justify-content: space-around; text-align: center; color: white; font-size: 13px;">
                                                    <div>⚡ Tiền xử lý: <b>{spd.get('preprocess', 0)} ms</b></div>
                                                    <div>🤖 Suy luận AI: <b>{spd.get('inference', 0)} ms</b></div>
                                                    <div>🎯 Hậu xử lý: <b>{spd.get('postprocess', 0)} ms</b></div>
                                                    <div>⏱️ Tổng thời gian: <b style="color: #00ff88;">{spd.get('total', 0)} ms ({spd.get('fps', 0)} FPS)</b></div>
                                                </div>
                                                """, unsafe_allow_html=True)
                                    except Exception:
                                        pass
                                    bytes_buf = bytes()
                                    break
                                else:
                                    break
                            else:
                                break
                else:
                    live_cam_status.error(f"❌ Không thể mở stream webcam: {resp_cam.text}")
            except Exception as e:
                live_cam_status.error(f"❌ Lỗi kết nối tới Backend khi stream webcam: {e}")
    else:
        st.markdown("Mở camera trực tiếp từ trình duyệt, đưa hoa lên trước ống kính và bấm **Take Photo** để YOLO 26 phân tích tức thì!")
        camera_photo = st.camera_input("Chụp ảnh từ Camera realtime", key="webcam_input")

        if camera_photo is not None:
            col_cam_in, col_cam_out = st.columns([1, 1])
            with col_cam_in:
                st.markdown("#### 📸 Ảnh vừa chụp từ Webcam")
                st.image(camera_photo, use_container_width=True)
                analyze_cam_btn = st.button("⚡ PHÂN TÍCH ẢNH WEBCAM NGAY", type="primary", use_container_width=True, key="btn_cam")

            with col_cam_out:
                st.markdown("#### 🎯 Kết quả suy luận YOLO 26")
                if analyze_cam_btn:
                    with st.spinner("⚡ YOLO 26 đang xử lý ảnh webcam..."):
                        try:
                            camera_photo.seek(0)
                            files = {"file": ("webcam_snap.jpg", camera_photo.getvalue(), "image/jpeg")}
                            data = {
                                "model_key": model_key,
                                "conf_threshold": str(conf_threshold),
                                "iou_threshold": str(iou_threshold),
                                "task_mode": task_mode,
                                "cls_model_key": cls_model_key,
                                "crop_padding": str(crop_padding),
                                "imgsz": str(imgsz),
                                "min_box_area": str(min_box_area)
                            }
                            response = requests.post(DETECT_URL, files=files, data=data)

                            if response.status_code == 200:
                                cam_json = response.json()
                                st.session_state["cam_result"] = cam_json
                            else:
                                st.error(f"❌ Lỗi từ Server: {response.text}")
                        except Exception as e:
                            st.error(f"❌ Lỗi kết nối tới Backend: {e}")

                if "cam_result" in st.session_state:
                    c_res = st.session_state["cam_result"]
                    if "image_base64" in c_res:
                        c_img_data = c_res["image_base64"].split(",")[1]
                        c_img_bytes = base64.b64decode(c_img_data)
                        st.image(Image.open(io.BytesIO(c_img_bytes)), use_container_width=True)

                        spd = c_res.get("speed_metrics", {})
                        if spd:
                            st.markdown(f"""
                            <div style="background: #1e1e24; padding: 10px 14px; border-radius: 10px; margin-top: 10px; border: 1px solid #333; display: flex; justify-content: space-around; text-align: center; color: white;">
                                <div><span style="font-size: 12px; color: #aaa;">⚡ Tiền xử lý</span><br><b>{spd.get('preprocess', 0)} ms</b></div>
                                <div><span style="font-size: 12px; color: #aaa;">🤖 Suy luận AI</span><br><b>{spd.get('inference', 0)} ms</b></div>
                                <div><span style="font-size: 12px; color: #aaa;">🎯 Hậu xử lý</span><br><b>{spd.get('postprocess', 0)} ms</b></div>
                                <div><span style="font-size: 12px; color: #aaa;">⏱️ Tổng thời gian</span><br><b style="color: #00ff88;">{spd.get('total', 0)} ms ({spd.get('fps', 0)} FPS)</b></div>
                            </div>
                            """, unsafe_allow_html=True)

            if "cam_result" in st.session_state:
                c_res = st.session_state["cam_result"]
                st.markdown("---")
                st.markdown("### 📚 Tra Cứu Thông Tin & Live Counter (SQL Server)")

                if c_res["task"] in ["detect", "hybrid"]:
                    items = c_res.get("detected_items", [])
                    if len(items) > 0:
                        counter = Counter([item.get("db_info", {}).get("name_vi", item["folder_name"]) for item in items])
                        pills = " | ".join([f"**{k}**: `{v}`" for k, v in counter.items()])
                        st.markdown(
                            f'<div style="padding: 0.6rem 1rem; background: rgba(46, 204, 113, 0.15); border-left: 4px solid #2ecc71; border-radius: 8px; margin-bottom: 1rem;">📊 **Live Flower Counter**: {pills}</div>',
                            unsafe_allow_html=True
                        )
                        st.success(f"🎉 Webcam phát hiện được **{len(items)}** bông hoa!")
                        for idx, item in enumerate(items):
                            db_info = item.get("db_info", {})
                            name_vi = db_info.get("name_vi", item["folder_name"])
                            desc = db_info.get("description", "Chưa có thông tin mô tả chi tiết.")
                            conf_cls_pct = item["confidence"] * 100
                            conf_det_pct = item.get("conf_det", item["confidence"]) * 100

                            if c_res["task"] == "hybrid" and item.get("conf_det") is not None:
                                title_str = f"🌺 #{idx + 1}: {name_vi} | 🎯 Det: {conf_det_pct:.1f}%  |  🏷️ Cls: {conf_cls_pct:.1f}%"
                            else:
                                title_str = f"🌺 #{idx + 1}: {name_vi} (Độ chính xác Detection: {conf_det_pct:.1f}%)"

                            with st.expander(title_str, expanded=True):
                                st.markdown(f"**Tên khoa học / Folder**: `{item['folder_name']}`")
                                st.markdown(f"**Mô tả từ Database**:\n> {desc}")
                    else:
                        st.warning("⚠️ Không nhận diện được loài hoa nào từ ảnh webcam chụp. Hãy thử gần hơn và đủ ánh sáng!")
                else:
                    top1 = c_res.get("top1")
                    if top1:
                        db_info = top1.get("db_info", {})
                        name_vi = db_info.get("name_vi", top1["folder_name"])
                        desc = db_info.get("description", "Chưa có mô tả chi tiết.")
                        conf_pct = top1["confidence"] * 100

                        st.success(f"🎉 Webcam phân loại hoa: **{name_vi}** (Độ chính xác: **{conf_pct:.2f}%**)")
                        st.markdown(f"> {desc}")