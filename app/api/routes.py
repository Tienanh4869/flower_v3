import os
import io
import base64
import tempfile
import traceback
from PIL import Image
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from app.services.ai_service import ai_service
from app.core.config import MODELS_CONFIG

router = APIRouter()


@router.get("/models")
def get_available_models():
    """
    Trả về danh sách 4 mô hình YOLO 26 cấu hình và trạng thái bộ nhớ RAM
    """
    return {
        "status": "success",
        "models": MODELS_CONFIG,
        "loaded_in_ram": list(ai_service.loaded_models.keys())
    }


@router.post("/detect")
async def detect_flower(
    file: UploadFile = File(...),
    model_key: str = Form("yolo26n_detect"),
    conf_threshold: float = Form(0.4),
    iou_threshold: float = Form(0.45),
    task_mode: str = Form("detect"),
    cls_model_key: str = Form("yolo26s_cls"),
    crop_padding: float = Form(0.05),
    imgsz: int = Form(640),
    min_box_area: float = Form(0.0)
):
    print(f"\n================= BẮT ĐẦU SUY LUẬN ({model_key} | task_mode={task_mode} | imgsz={imgsz}) ==================")
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if task_mode == "hybrid":
            output = ai_service.predict_hybrid(
                image,
                det_model_key=model_key,
                cls_model_key=cls_model_key,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                crop_padding=crop_padding,
                imgsz=imgsz,
                min_box_area=min_box_area
            )
        else:
            output = ai_service.predict_and_get_info(
                image,
                model_key=model_key,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                imgsz=imgsz,
                min_box_area=min_box_area
            )

        if "error" in output:
            print(f"❌ Lỗi từ ai_service trả về: {output['error']}")
            raise HTTPException(status_code=500, detail=output["error"])

        buffered = io.BytesIO()
        output["plotted_image"].save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        task = output["task"]
        print(f"✅ SUY LUẬN THÀNH CÔNG ({task.upper()})!")

        if task == "detect" or task == "hybrid":
            return {
                "status": "success",
                "task": task,
                "image_base64": f"data:image/jpeg;base64,{img_base64}",
                "detected_items": output.get("detected_items", []),
                "speed_metrics": output.get("speed_metrics", {})
            }
        else:
            return {
                "status": "success",
                "task": task,
                "image_base64": f"data:image/jpeg;base64,{img_base64}",
                "top1": output.get("top1"),
                "top5": output.get("top5"),
                "speed_metrics": output.get("speed_metrics", {})
            }
    except HTTPException:
        raise
    except Exception as e:
        print("❌ LỖI CRASH THẬT SỰ:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi thật: {str(e)}")


@router.post("/detect_video")
async def detect_video(
    file: UploadFile = File(...),
    model_key: str = Form("yolo26n_detect"),
    conf_threshold: float = Form(0.4),
    iou_threshold: float = Form(0.45),
    task_mode: str = Form("detect"),
    cls_model_key: str = Form("yolo26s_cls"),
    crop_padding: float = Form(0.05),
    imgsz: int = Form(640),
    min_box_area: float = Form(0.0),
    skip_frames: int = Form(1)
):
    print(f"\n================= BẮT ĐẦU SUY LUẬN VIDEO ({model_key} | {task_mode} | skip={skip_frames}) ==================")
    temp_in_path = None
    temp_out_path = None
    try:
        video_bytes = await file.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(video_bytes)
            temp_in_path = tmp_in.name

        temp_out_path = temp_in_path.replace(".mp4", "_out.mp4")

        output = ai_service.predict_video(
            video_path=temp_in_path,
            output_path=temp_out_path,
            model_key=model_key,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            task_mode=task_mode,
            cls_model_key=cls_model_key,
            crop_padding=crop_padding,
            imgsz=imgsz,
            min_box_area=min_box_area,
            skip_frames=skip_frames
        )

        if "error" in output:
            raise HTTPException(status_code=500, detail=output["error"])

        with open(temp_out_path, "rb") as f:
            out_bytes = f.read()
        
        video_base64 = base64.b64encode(out_bytes).decode("utf-8")

        print("✅ XỬ LÝ VIDEO THÀNH CÔNG!")
        return {
            "status": "success",
            "task": output["task"],
            "total_frames": output["total_frames"],
            "summary": output["summary"],
            "video_base64": f"data:video/mp4;base64,{video_base64}"
        }
    except HTTPException:
        raise
    except Exception as e:
        print("❌ LỖI CRASH VIDEO:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi thật: {str(e)}")
    finally:
        if temp_in_path and os.path.exists(temp_in_path):
            try:
                os.remove(temp_in_path)
            except Exception:
                pass
        if temp_out_path and os.path.exists(temp_out_path):
            try:
                os.remove(temp_out_path)
            except Exception:
                pass


@router.post("/detect_video_stream")
async def detect_video_stream(
    file: UploadFile = File(...),
    model_key: str = Form("yolo26n_detect"),
    conf_threshold: float = Form(0.4),
    iou_threshold: float = Form(0.45),
    skip_frames: int = Form(2),
    task_mode: str = Form("detect"),
    cls_model_key: str = Form("yolo26s_cls"),
    crop_padding: float = Form(0.05),
    imgsz: int = Form(640),
    min_box_area: float = Form(0.0)
):
    print(f"\n================= BẮT ĐẦU LIVE STREAM VIDEO ({model_key} | task_mode={task_mode} | skip={skip_frames}) ==================")
    video_bytes = await file.read()

    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_in.write(video_bytes)
    tmp_in.close()
    temp_in_path = tmp_in.name

    def stream_and_cleanup():
        try:
            for chunk in ai_service.generate_video_frames(
                video_path=temp_in_path,
                model_key=model_key,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                skip_frames=skip_frames,
                task_mode=task_mode,
                cls_model_key=cls_model_key,
                crop_padding=crop_padding,
                imgsz=imgsz,
                min_box_area=min_box_area
            ):
                yield chunk
        finally:
            if os.path.exists(temp_in_path):
                try:
                    os.remove(temp_in_path)
                except Exception:
                    pass

    return StreamingResponse(
        stream_and_cleanup(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/detect_webcam_stream")
def detect_webcam_stream(
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
    print(f"\n================= BẮT ĐẦU LIVE STREAM WEBCAM REALTIME ({model_key} | task_mode={task_mode} | cam={camera_index}) ==================")
    return StreamingResponse(
        ai_service.generate_webcam_frames(
            model_key=model_key,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            skip_frames=skip_frames,
            task_mode=task_mode,
            cls_model_key=cls_model_key,
            crop_padding=crop_padding,
            imgsz=imgsz,
            min_box_area=min_box_area,
            camera_index=camera_index
        ),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )