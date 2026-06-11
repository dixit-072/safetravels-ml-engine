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
    # Extract dynamic network configurations assigned by Render
    # If local, defaults flawlessly back to 127.0.0.1:8000
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    
    uvicorn.run("backend.main:app", host=host, port=port, reload=True)