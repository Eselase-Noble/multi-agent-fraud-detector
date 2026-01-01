from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from typing import Dict, Any

from app.api.routes import router as api_router
from app.utils.config import settings
from app.utils.logger import setup_logger
from app.db.postgres import init_db, close_db
from app.db.vector_store import init_vector_store

#author: Noble Eselase Vulley
#version: 1.0.0

# Setup logger
logger = setup_logger(__name__)
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    logger.info("Starting IntelliFraud Copilot...")
    try:
        # Initialize databases
        await init_db()
        await init_vector_store()
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down IntelliFraud Copilot...")
    await close_db()
    logger.info("Services closed successfully")



# Create FastAPI app
app = FastAPI(
    title="IntelliFraud Copilot API",
    description="Multi-Agent RAG System for Financial Fraud Analysis",
    version="1.0.0",
    lifespan=lifespan
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")


# Health check endpoint
@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "IntelliFraud Copilot",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    try:
        # Check database
        from app.db.postgres import db_manager
        with db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")

        # Check vector store
        from app.db.vector_store import vector_store
        if not vector_store.is_ready():
            raise Exception("Vector store not ready")

        return {
            "status": "healthy",
            "database": "connected",
            "vector_store": "ready",
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")



# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "request_id": request.state.request_id if hasattr(request.state, 'request_id') else None
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )