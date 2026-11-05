"""Main FastAPI application entrypoint with lifespan model management."""

from contextlib import asynccontextmanager
import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import router as api_router
from app.config import APP_NAME, VERSION
from models.manager import DetectorManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager: starts server instantly with on-demand model loading."""
    logger.info("Initializing Object Detector & Counter...")
    manager = DetectorManager()
    app.state.detector_manager = manager
    logger.info("Server ready to accept requests. Model weights load on-demand.")

    yield

    logger.info("Shutting down Object Detector & Counter...")


# Create FastAPI instance
app = FastAPI(
    title=APP_NAME,
    version=VERSION,
    description="Real-time object detection and counting web API using YOLOS, RT-DETR, and OWLv2.",
    lifespan=lifespan,
)

# Enable Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

# Mount static folder
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve the single-page frontend application."""
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": f"{APP_NAME} v{VERSION} is running. Access /docs for API documentation."}

