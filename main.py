from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

app = FastAPI(
    title="🌸 Flower AI Recognition & Knowledge API",
    description="Hệ thống API nhận diện hoa bằng YOLO và tra cứu SQL Server",
    version="1.0.0"
)

# Cho phép Frontend kết nối (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký router
app.include_router(router, prefix="/api/v1")

@router.get("/")
def home():
    return {"message": "🌸 Welcome to Flower Recognition API! Truy cập /docs để xem Swagger UI."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)