from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from typing import Dict, Any, Optional
import logging
from dotenv import load_dotenv
import base64
from dotenv import load_dotenv

load_dotenv()

from image_processing import ImageProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Media Context Processor", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Initialize processor
image_processor = ImageProcessor(
    serp_api_key=os.getenv("SERP_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

@app.post("/analyze-url/")
async def analyze_image_url(image_url: str = Query(..., description="Image URL to analyze")) -> Dict[str, Any]:
    """Analyze image from URL - main method"""
    try:
        result = image_processor.get_google_lens_context(image_url)
        return {
            "image_url": image_url,
            "status": "success",
            "lens_context": result
        }
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "features": {
            "serp_api": bool(os.getenv("SERP_API_KEY")),
            "openai": bool(os.getenv("OPENAI_API_KEY"))
        }
    }

@app.get("/")
async def root():
    return {
        "message": "Media Context Processor with Enhanced BOOM Fact-Check",
        "endpoints": {
            "analyze_url": "/analyze-url/",
            "health": "/health"
        },
        "usage": "Use /analyze-url/ with image_url parameter to analyze images"
    }

if __name__ == "__main__":
    required_vars = ["SERP_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        logger.error(f"Missing: {missing}")
        print("Required environment variables:")
        print("  SERP_API_KEY (required)")
        print("  OPENAI_API_KEY (optional, for better context)")
        exit(1)
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)