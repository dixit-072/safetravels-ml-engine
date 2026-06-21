import sys
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # 👈 Added security middleware

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.api.routes import router

app = FastAPI(
    title="SafeTravels ML Risk Engine Core",
    version="2.0.6"
)

# 🛡️ GLOBAL CORS DISPATCHER: Allows your Streamlit site to talk to your Render server securely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permits all incoming web origins to talk to your models
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🟢 KEEP-ALIVE ENDPOINT: Explicitly handles GET and HEAD requests to prevent 405 errors from UptimeRobot
@app.get("/health")
@app.head("/health")
async def health_check():
    return {"status": "ok", "message": "Engine is awake and operational"}

app.include_router(router)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    uvicorn.run("backend.main:app", host=host, port=port, reload=True)