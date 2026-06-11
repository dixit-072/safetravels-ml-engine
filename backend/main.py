import sys
import os
import uvicorn
from fastapi import FastAPI

# Explicitly append root path to bypass internal module resolution drops
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.api.routes import router

app = FastAPI(
    title="SafeTravels ML Risk Engine Core",
    version="2.0.6",
    description="Enterprise-grade decoupled route hazard predictive microservice."
)

# Bind modular endpoints router
app.include_router(router)

if __name__ == "__main__":
    # FIX: Removed the '.py' so uvicorn resolves the module paths seamlessly
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)