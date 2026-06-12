from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.health import router as health_router
from routes.prediction import router as prediction_router
from routes.camera import router as camera_router
from routes.history import router as history_router
from routes.analytics import router as analytics_router
from routes.export import router as export_router

app = FastAPI(
    title="PharmaCheck API",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(prediction_router)
app.include_router(camera_router)
app.include_router(history_router)
app.include_router(analytics_router)
app.include_router(export_router)