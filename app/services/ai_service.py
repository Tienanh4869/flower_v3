import os
import io
import json
import base64
import cv2
import numpy as np
import imageio
from PIL import Image
from ultralytics import YOLO
from app.core.config import MODELS_CONFIG, MODEL_DETECT_PATH
from app.services.db_service import get_flower_info_by_name


def draw_hud_overlay(frame: np.ndarray, frame_count: int, total_frames: int, video_fps: float, speed_metrics: dict, active_count: int = 0) -> np.ndarray:
    """Vẽ HUD Telemetry chuẩn khoa học lên góc trên bên trái của frame video/webcam."""
    if frame is None or not isinstance(frame, np.ndarray):
        return frame
    h, w = frame.shape[:2]
    video_fps = video_fps if video_fps > 0 else 25.0
    sec = int(frame_count / video_fps)
    ms = int(((frame_count / video_fps) - sec) * 100)
    
    if total_frames > 0:
        line1 = f"Frame: #{frame_count}/{total_frames} | Time: {sec:02d}.{ms:02d}s"
    else:
        line1 = f"Live Frame: #{frame_count} | Time: {sec:02d}s"
    
    fps_val = speed_metrics.get("fps", 0.0)
    lat_val = speed_metrics.get("total", 0.0)
    line2 = f"Speed: {fps_val} FPS ({lat_val} ms)"
    line3 = f"Active Detections: {active_count} flowers"
    
    scale = max(0.55, min(0.9, w / 1280.0))
    thickness = max(1, int(scale * 2))
    
    lines = [line1, line2, line3]
    max_tw = 0
    for l in lines:
        (tw, th), _ = cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if tw > max_tw:
            max_tw = tw
    
    pad_x, pad_y = 15, 12
    box_w = max_tw + pad_x * 2
    line_spacing = int(24 * (scale / 0.6))
    box_h = line_spacing * len(lines) + pad_y * 2
    
    # Nền tối viền xanh công nghệ
    cv2.rectangle(frame, (10, 10), (10 + box_w, 10 + box_h), (25, 25, 35), -1)
    cv2.rectangle(frame, (10, 10), (10 + box_w, 10 + box_h), (0, 220, 150), 2)
    
    # Vẽ từng dòng
    y_curr = 10 + pad_y + int(18 * (scale / 0.6))
    cv2.putText(frame, line1, (10 + pad_x, y_curr), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    y_curr += line_spacing
    cv2.putText(frame, line2, (10 + pad_x, y_curr), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 180), thickness, cv2.LINE_AA)
    y_curr += line_spacing
    cv2.putText(frame, line3, (10 + pad_x, y_curr), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 190, 80), thickness, cv2.LINE_AA)
    return frame


class FlowerAIService:
    def __init__(self):
        self.loaded_models = {}
        # Pre-load default model
        self.get_or_load_model("yolo26n_detect")

    def get_or_load_model(self, model_key: str = "yolo26n_detect"):
        if model_key not in MODELS_CONFIG:
            model_key = "yolo26n_detect"
        
        cfg = MODELS_CONFIG[model_key]
        path = cfg["path"]
        
        if model_key not in self.loaded_models:
            if os.path.exists(path):
                self.loaded_models[model_key] = YOLO(path)
                print(f"✅ [MODEL CACHE] Đã tải thành công model vào RAM: {model_key} ({path})")
            else:
                print(f"⚠️ [MODEL CACHE] Không tìm thấy file model tại: {path}")
                return None, cfg
        return self.loaded_models.get(model_key), cfg

    def predict_and_get_info(self, pil_image: Image.Image, model_key: str = "yolo26n_detect", conf_threshold: float = 0.4, iou_threshold: float = 0.45, imgsz: int = 640, min_box_area: float = 0.0):
        """
        Dự đoán ảnh với model được chọn (Detection hoặc Classification) và tra cứu SQL Server
        """
        model, cfg = self.get_or_load_model(model_key)
        if model is None:
            return {"error": f"Model AI chưa được tải! Không tìm thấy file tại: {cfg['path']}"}

        task = cfg["task"]

        # 1. Chạy suy luận YOLO
        if task == "detect":
            results = model(pil_image, conf=conf_threshold, iou=iou_threshold, imgsz=imgsz, verbose=False)
        else:
            results = model(pil_image, conf=conf_threshold, imgsz=imgsz, verbose=False)
        
        result = results[0]

        # Lấy thống kê tốc độ (Latency Breakdown Metrics - ms)
        speed_dict = getattr(result, "speed", {})
        prep_ms = round(float(speed_dict.get("preprocess", 0.0)), 2)
        inf_ms = round(float(speed_dict.get("inference", 0.0)), 2)
        post_ms = round(float(speed_dict.get("postprocess", 0.0)), 2)
        tot_ms = round(prep_ms + inf_ms + post_ms, 2)
        fps = round(1000.0 / tot_ms, 1) if tot_ms > 0 else 0.0
        speed_metrics = {
            "preprocess": prep_ms,
            "inference": inf_ms,
            "postprocess": post_ms,
            "total": tot_ms,
            "fps": fps
        }

        # 2. Xử lý ảnh kết quả có vẽ sẵn Bounding Box / Nhãn phân loại với viền dày rõ nét
        res_plotted = result.plot(line_width=3)  # numpy array (BGR)
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        plotted_image = Image.fromarray(res_rgb)

        img_area = float(pil_image.width * pil_image.height)

        if task == "detect":
            detected_items = []
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    box_area_pct = ((x2 - x1) * (y2 - y1) / img_area) * 100.0
                    if box_area_pct < min_box_area:
                        continue

                    class_id = int(box.cls[0].item())
                    confidence = round(float(box.conf[0].item()), 4)
                    folder_name = model.names[class_id]
                    db_info = get_flower_info_by_name(folder_name)

                    detected_items.append({
                        "class_id": class_id,
                        "folder_name": folder_name,
                        "confidence": confidence,
                        "conf_det": confidence,
                        "box_coords": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                        "box_area_pct": round(box_area_pct, 2),
                        "db_info": db_info if db_info else {
                            "name_vi": f"Chưa có data cho: {folder_name}",
                            "description": "Vui lòng cập nhật thêm vào SQL Server."
                        }
                    })
            return {
                "task": "detect",
                "plotted_image": plotted_image,
                "detected_items": detected_items,
                "speed_metrics": speed_metrics
            }
        
        elif task == "cls":
            probs = result.probs
            top1_info = None
            top5_list = []
            if probs is not None:
                top1_id = int(probs.top1)
                top1_conf = round(float(probs.top1conf), 4)
                folder_name = model.names[top1_id]
                db_info = get_flower_info_by_name(folder_name)
                top1_info = {
                    "class_id": top1_id,
                    "folder_name": folder_name,
                    "confidence": top1_conf,
                    "db_info": db_info if db_info else {
                        "name_vi": f"Chưa có data cho: {folder_name}",
                        "description": "Vui lòng cập nhật thêm vào SQL Server."
                    }
                }

                # Top 5 details
                for cls_id, conf in zip(probs.top5, probs.top5conf.tolist()):
                    cls_id_int = int(cls_id)
                    fname = model.names[cls_id_int]
                    db_inf = get_flower_info_by_name(fname)
                    top5_list.append({
                        "class_id": cls_id_int,
                        "folder_name": fname,
                        "confidence": round(float(conf), 4),
                        "db_info": db_inf if db_inf else {"name_vi": fname, "description": ""}
                    })

            return {
                "task": "cls",
                "plotted_image": plotted_image,
                "top1": top1_info,
                "top5": top5_list,
                "speed_metrics": speed_metrics
            }

    def predict_hybrid(
        self,
        pil_image: Image.Image,
        det_model_key: str = "yolo26n_flower_only",
        cls_model_key: str = "yolo26s_cls",
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        crop_padding: float = 0.05,
        imgsz: int = 640,
        min_box_area: float = 0.0
    ):
        """
        Two-Stage Hybrid: Stage 1 Detect hoa -> Stage 2 Crop + Classify từng bông -> Trả về Gallery Base64 & Top-3
        """
        det_model, det_cfg = self.get_or_load_model(det_model_key)
        cls_model, cls_cfg = self.get_or_load_model(cls_model_key)

        if det_model is None:
            return {"error": f"Model Detection chưa được tải! Không tìm thấy file: {det_cfg['path']}"}
        if cls_model is None:
            return {"error": f"Model Classification chưa được tải! Không tìm thấy file: {cls_cfg['path']}"}

        det_results = det_model(pil_image, conf=conf_threshold, iou=iou_threshold, imgsz=imgsz, verbose=False)
        boxes = det_results[0].boxes

        speed_dict = getattr(det_results[0], "speed", {})
        prep_ms = round(float(speed_dict.get("preprocess", 0.0)), 2)
        det_inf_ms = float(speed_dict.get("inference", 0.0))
        post_ms = round(float(speed_dict.get("postprocess", 0.0)), 2)
        cls_inf_total = 0.0

        detected_items = []
        img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        img_area = float(pil_image.width * pil_image.height)

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                box_w = x2 - x1
                box_h = y2 - y1
                box_area_pct = ((box_w * box_h) / img_area) * 100.0
                if box_area_pct < min_box_area:
                    continue

                conf_det = round(float(box.conf[0].item()), 4)

                # Mở rộng bounding box theo tỷ lệ crop_padding
                pad_w = box_w * crop_padding
                pad_h = box_h * crop_padding
                x1_p = max(0, int(x1 - pad_w))
                y1_p = max(0, int(y1 - pad_h))
                x2_p = min(pil_image.width, int(x2 + pad_w))
                y2_p = min(pil_image.height, int(y2 + pad_h))

                # Cắt ảnh bông hoa
                cropped_pil = pil_image.crop((x1_p, y1_p, x2_p, y2_p))

                # Chuyển đổi sang chuỗi base64 cho Frontend hiển thị Gallery
                buf = io.BytesIO()
                cropped_pil.save(buf, format="JPEG")
                crop_base64 = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

                # Phân loại bông hoa đã crop bằng cls_model
                cls_results = cls_model(cropped_pil, imgsz=imgsz, verbose=False)[0]
                cls_inf_total += float(getattr(cls_results, "speed", {}).get("inference", 0.0))

                probs = cls_results.probs
                top1_id = int(probs.top1)
                conf_cls = round(float(probs.top1conf), 4)
                folder_name = cls_model.names[top1_id]

                db_info = get_flower_info_by_name(folder_name)

                top3_list = []
                if probs.top5 is not None:
                    for cid, cconf in zip(probs.top5[:3], probs.top5conf[:3].tolist()):
                        c_int = int(cid)
                        fname = cls_model.names[c_int]
                        db_inf = get_flower_info_by_name(fname)
                        top3_list.append({
                            "folder_name": fname,
                            "confidence": round(float(cconf), 4),
                            "name_vi": db_inf["name_vi"] if db_inf else fname
                        })

                detected_items.append({
                    "class_id": top1_id,
                    "folder_name": folder_name,
                    "confidence": conf_cls,
                    "conf_det": conf_det,
                    "conf_cls": conf_cls,
                    "box_coords": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "box_area_pct": round(box_area_pct, 2),
                    "crop_base64": crop_base64,
                    "db_info": db_info if db_info else {
                        "name_vi": f"Chưa có data cho: {folder_name}",
                        "description": "Vui lòng cập nhật thêm vào SQL Server."
                    },
                    "top3": top3_list
                })

                # Vẽ khung và nhãn loài lên ảnh tổng hợp viền dày chữ to rõ nét
                name_vi = db_info["name_vi"] if db_info else folder_name
                label = f"{name_vi} | Det:{conf_det*100:.0f}% Cls:{conf_cls*100:.0f}%"
                x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
                color = (140, 117, 255)
                cv2.rectangle(img_np, (x1_i, y1_i), (x2_i, y2_i), color, 3)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                cv2.rectangle(img_np, (x1_i, y1_i - th - 10), (x1_i + tw + 6, y1_i), color, -1)
                cv2.putText(img_np, label, (x1_i + 3, y1_i - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        tot_inf_ms = round(det_inf_ms + cls_inf_total, 2)
        tot_ms = round(prep_ms + tot_inf_ms + post_ms, 2)
        fps = round(1000.0 / tot_ms, 1) if tot_ms > 0 else 0.0
        speed_metrics = {
            "preprocess": prep_ms,
            "inference": tot_inf_ms,
            "postprocess": post_ms,
            "total": tot_ms,
            "fps": fps
        }

        plotted_image = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        return {
            "task": "hybrid",
            "plotted_image": plotted_image,
            "detected_items": detected_items,
            "speed_metrics": speed_metrics
        }

    def predict_video(
        self,
        video_path: str,
        output_path: str,
        model_key: str = "yolo26n_detect",
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        task_mode: str = "detect",
        cls_model_key: str = "yolo26s_cls",
        crop_padding: float = 0.05,
        imgsz: int = 640,
        min_box_area: float = 0.0
    ):
        """
        Dự đoán video frame-by-frame, ghi video kết quả và tổng hợp thống kê
        """
        if task_mode != "hybrid":
            model, cfg = self.get_or_load_model(model_key)
            if model is None:
                return {"error": f"Model AI chưa được tải! Không tìm thấy file tại: {cfg['path']}"}
            task = cfg["task"]
        else:
            task = "hybrid"

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Không thể mở file video đầu vào."}

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        try:
            writer = imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
            use_imageio = True
        except Exception as e:
            print(f"⚠️ Không dùng được imageio ({e}), fallback qua OpenCV VideoWriter")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            use_imageio = False

        summary_counts = {}
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            if task_mode == "hybrid":
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                hybrid_res = self.predict_hybrid(
                    pil_frame,
                    det_model_key=model_key,
                    cls_model_key=cls_model_key,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    crop_padding=crop_padding,
                    imgsz=imgsz,
                    min_box_area=min_box_area
                )
                if "error" in hybrid_res:
                    plotted = frame
                else:
                    plotted = cv2.cvtColor(np.array(hybrid_res["plotted_image"]), cv2.COLOR_RGB2BGR)
                    for item in hybrid_res.get("detected_items", []):
                        fname = item["folder_name"]
                        summary_counts[fname] = summary_counts.get(fname, 0) + 1
            else:
                if task == "detect":
                    results = model(frame, conf=conf_threshold, iou=iou_threshold, imgsz=imgsz, verbose=False)
                else:
                    results = model(frame, conf=conf_threshold, imgsz=imgsz, verbose=False)

                res = results[0]
                plotted = res.plot(line_width=3)

                if task == "detect" and res.boxes is not None:
                    img_area = float(frame.shape[0] * frame.shape[1])
                    for box in res.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        if ((x2 - x1) * (y2 - y1) / img_area) * 100.0 < min_box_area:
                            continue
                        cls_id = int(box.cls[0].item())
                        name = model.names[cls_id]
                        summary_counts[name] = summary_counts.get(name, 0) + 1
                elif task == "cls" and res.probs is not None:
                    top1_id = int(res.probs.top1)
                    name = model.names[top1_id]
                    summary_counts[name] = summary_counts.get(name, 0) + 1

            active_cnt = 0
            speed_m = {}
            if task_mode == "hybrid":
                if "error" not in hybrid_res:
                    active_cnt = len(hybrid_res.get("detected_items", []))
                    speed_m = hybrid_res.get("speed_metrics", {})
            else:
                if task == "detect" and res.boxes is not None:
                    active_cnt = len(res.boxes)
                elif task == "cls" and res.probs is not None:
                    active_cnt = 1
                speed_dict = getattr(res, "speed", {})
                prep = round(float(speed_dict.get("preprocess", 0.0)), 2)
                inf = round(float(speed_dict.get("inference", 0.0)), 2)
                post = round(float(speed_dict.get("postprocess", 0.0)), 2)
                tot = round(prep + inf + post, 2)
                speed_m = {"fps": round(1000.0 / tot, 1) if tot > 0 else 0.0, "total": tot}

            plotted = draw_hud_overlay(plotted, frame_count, total_frames, fps, speed_m, active_cnt)

            if use_imageio:
                rgb_plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
                writer.append_data(rgb_plotted)
            else:
                out.write(plotted)

        cap.release()
        if use_imageio:
            writer.close()
        else:
            out.release()

        detailed_summary = []
        for fname, count in summary_counts.items():
            db_info = get_flower_info_by_name(fname)
            name_vi = db_info["name_vi"] if db_info else fname
            detailed_summary.append({
                "folder_name": fname,
                "name_vi": name_vi,
                "occurrences": count
            })

        return {
            "status": "success",
            "task": task,
            "total_frames": frame_count,
            "summary": detailed_summary
        }

    def generate_video_frames(
        self,
        video_path: str,
        model_key: str = "yolo26n_detect",
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        skip_frames: int = 1,
        task_mode: str = "detect",
        cls_model_key: str = "yolo26s_cls",
        crop_padding: float = 0.05,
        imgsz: int = 640,
        min_box_area: float = 0.0
    ):
        """
        Tạo luồng generator MJPEG frame-by-frame giúp hiển thị trực tiếp lên web với độ trễ ~0s (Live Streaming).
        """
        if skip_frames < 1:
            skip_frames = 1

        if task_mode != "hybrid":
            model, cfg = self.get_or_load_model(model_key)
            if model is None:
                error_json = json.dumps({"error": f"Model AI chưa được tải: {cfg['path']}"}).encode("utf-8")
                yield (b'--frame\r\n'
                       b'Content-Type: application/json\r\n\r\n' + error_json + b'\r\n')
                return
            task = cfg["task"]
        else:
            task = "hybrid"

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error_json = json.dumps({"error": "Không thể mở file video đầu vào."}).encode("utf-8")
            yield (b'--frame\r\n'
                   b'Content-Type: application/json\r\n\r\n' + error_json + b'\r\n')
            return

        summary_counts = {}
        frame_count = 0
        last_plotted = None
        last_speed = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            if frame_count != 1 and frame_count % skip_frames != 0:
                continue

            if task_mode == "hybrid":
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                hybrid_res = self.predict_hybrid(
                    pil_frame,
                    det_model_key=model_key,
                    cls_model_key=cls_model_key,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    crop_padding=crop_padding,
                    imgsz=imgsz,
                    min_box_area=min_box_area
                )
                if "error" in hybrid_res:
                    last_plotted = frame
                else:
                    last_plotted = cv2.cvtColor(np.array(hybrid_res["plotted_image"]), cv2.COLOR_RGB2BGR)
                    last_speed = hybrid_res.get("speed_metrics", {})
                    for item in hybrid_res.get("detected_items", []):
                        fname = item["folder_name"]
                        summary_counts[fname] = summary_counts.get(fname, 0) + 1
            else:
                if task == "detect":
                    results = model(frame, conf=conf_threshold, iou=iou_threshold, imgsz=imgsz, verbose=False)
                else:
                    results = model(frame, conf=conf_threshold, imgsz=imgsz, verbose=False)

                res = results[0]
                last_plotted = res.plot(line_width=3)
                speed_dict = getattr(res, "speed", {})
                prep = round(float(speed_dict.get("preprocess", 0.0)), 2)
                inf = round(float(speed_dict.get("inference", 0.0)), 2)
                post = round(float(speed_dict.get("postprocess", 0.0)), 2)
                tot = round(prep + inf + post, 2)
                last_speed = {
                    "preprocess": prep,
                    "inference": inf,
                    "postprocess": post,
                    "total": tot,
                    "fps": round(1000.0 / tot, 1) if tot > 0 else 0.0
                }

                if task == "detect" and res.boxes is not None:
                    img_area = float(frame.shape[0] * frame.shape[1])
                    for box in res.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        if ((x2 - x1) * (y2 - y1) / img_area) * 100.0 < min_box_area:
                            continue
                        cls_id = int(box.cls[0].item())
                        name = model.names[cls_id]
                        summary_counts[name] = summary_counts.get(name, 0) + 1
                elif task == "cls" and res.probs is not None:
                    top1_id = int(res.probs.top1)
                    name = model.names[top1_id]
                    summary_counts[name] = summary_counts.get(name, 0) + 1

            active_cnt = 0
            if task_mode == "hybrid" and "hybrid_res" in locals() and isinstance(hybrid_res, dict):
                active_cnt = len(hybrid_res.get("detected_items", []))
            elif task_mode != "hybrid" and "res" in locals():
                if task == "detect" and getattr(res, "boxes", None) is not None:
                    active_cnt = len(res.boxes)
                elif task == "cls" and getattr(res, "probs", None) is not None:
                    active_cnt = 1

            hud_frame = draw_hud_overlay(last_plotted.copy() if last_plotted is not None else frame, frame_count, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS), last_speed, active_cnt)
            _, buffer = cv2.imencode('.jpg', hud_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            live_dict = {}
            for fname, count in summary_counts.items():
                db_inf = get_flower_info_by_name(fname)
                live_dict[db_inf["name_vi"] if db_inf else fname] = count
            live_payload = json.dumps({"live_counts": live_dict, "speed_metrics": last_speed}).encode("utf-8")
            yield (b'--frame\r\n'
                   b'Content-Type: application/json\r\n\r\n' + live_payload + b'\r\n')

        cap.release()

        detailed_summary = []
        for fname, count in summary_counts.items():
            db_info = get_flower_info_by_name(fname)
            name_vi = db_info["name_vi"] if db_info else fname
            detailed_summary.append({
                "folder_name": fname,
                "name_vi": name_vi,
                "occurrences": count
            })

        summary_payload = json.dumps({
            "status": "success",
            "task": task,
            "total_frames": frame_count,
            "summary": detailed_summary
        }).encode("utf-8")
        yield (b'--frame\r\n'
               b'Content-Type: application/json\r\n\r\n' + summary_payload + b'\r\n')

    def generate_webcam_frames(
        self,
        model_key: str = "yolo26n_detect",
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        skip_frames: int = 1,
        task_mode: str = "detect",
        cls_model_key: str = "yolo26s_cls",
        crop_padding: float = 0.05,
        imgsz: int = 640,
        min_box_area: float = 0.0,
        camera_index: int = 0
    ):
        """
        True Realtime Video Stream qua Webcam (30 FPS)
        """
        if skip_frames < 1:
            skip_frames = 1

        if task_mode != "hybrid":
            model, cfg = self.get_or_load_model(model_key)
            if model is None:
                error_json = json.dumps({"error": f"Model AI chưa được tải! Không tìm thấy file tại: {cfg['path']}"}).encode("utf-8")
                yield (b'--frame\r\n'
                       b'Content-Type: application/json\r\n\r\n' + error_json + b'\r\n')
                return
            task = cfg["task"]
        else:
            task = "hybrid"

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index + cv2.CAP_DSHOW)
            if not cap.isOpened():
                error_json = json.dumps({"error": "Không thể kết nối với Webcam trực tiếp từ máy (Camera Index 0)."}).encode("utf-8")
                yield (b'--frame\r\n'
                       b'Content-Type: application/json\r\n\r\n' + error_json + b'\r\n')
                return

        summary_counts = {}
        frame_count = 0
        last_plotted = None
        last_speed = {}

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                if frame_count != 1 and frame_count % skip_frames != 0:
                    continue

                if task_mode == "hybrid":
                    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    hybrid_res = self.predict_hybrid(
                        pil_frame,
                        det_model_key=model_key,
                        cls_model_key=cls_model_key,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                        crop_padding=crop_padding,
                        imgsz=imgsz,
                        min_box_area=min_box_area
                    )
                    if "error" in hybrid_res:
                        last_plotted = frame
                    else:
                        last_plotted = cv2.cvtColor(np.array(hybrid_res["plotted_image"]), cv2.COLOR_RGB2BGR)
                        last_speed = hybrid_res.get("speed_metrics", {})
                        for item in hybrid_res.get("detected_items", []):
                            fname = item["folder_name"]
                            summary_counts[fname] = summary_counts.get(fname, 0) + 1
                else:
                    if task == "detect":
                        results = model(frame, conf=conf_threshold, iou=iou_threshold, imgsz=imgsz, verbose=False)
                    else:
                        results = model(frame, conf=conf_threshold, imgsz=imgsz, verbose=False)

                    res = results[0]
                    last_plotted = res.plot(line_width=3)
                    speed_dict = getattr(res, "speed", {})
                    prep = round(float(speed_dict.get("preprocess", 0.0)), 2)
                    inf = round(float(speed_dict.get("inference", 0.0)), 2)
                    post = round(float(speed_dict.get("postprocess", 0.0)), 2)
                    tot = round(prep + inf + post, 2)
                    last_speed = {
                        "preprocess": prep,
                        "inference": inf,
                        "postprocess": post,
                        "total": tot,
                        "fps": round(1000.0 / tot, 1) if tot > 0 else 0.0
                    }

                    if task == "detect" and res.boxes is not None:
                        img_area = float(frame.shape[0] * frame.shape[1])
                        for box in res.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            if ((x2 - x1) * (y2 - y1) / img_area) * 100.0 < min_box_area:
                                continue
                            cls_id = int(box.cls[0].item())
                            name = model.names[cls_id]
                            summary_counts[name] = summary_counts.get(name, 0) + 1
                    elif task == "cls" and res.probs is not None:
                        top1_id = int(res.probs.top1)
                        name = model.names[top1_id]
                        summary_counts[name] = summary_counts.get(name, 0) + 1

                active_cnt = 0
                if task_mode == "hybrid" and "hybrid_res" in locals() and isinstance(hybrid_res, dict):
                    active_cnt = len(hybrid_res.get("detected_items", []))
                elif task_mode != "hybrid" and "res" in locals():
                    if task == "detect" and getattr(res, "boxes", None) is not None:
                        active_cnt = len(res.boxes)
                    elif task == "cls" and getattr(res, "probs", None) is not None:
                        active_cnt = 1

                hud_frame = draw_hud_overlay(last_plotted.copy() if last_plotted is not None else frame, frame_count, 0, cap.get(cv2.CAP_PROP_FPS), last_speed, active_cnt)
                _, buffer = cv2.imencode('.jpg', hud_frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

                live_dict = {}
                for fname, count in summary_counts.items():
                    db_inf = get_flower_info_by_name(fname)
                    live_dict[db_inf["name_vi"] if db_inf else fname] = count
                live_payload = json.dumps({"live_counts": live_dict, "speed_metrics": last_speed}).encode("utf-8")
                yield (b'--frame\r\n'
                       b'Content-Type: application/json\r\n\r\n' + live_payload + b'\r\n')
        finally:
            if cap.isOpened():
                cap.release()


# Khởi tạo instance service dùng chung
ai_service = FlowerAIService()