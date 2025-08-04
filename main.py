# from fastapi import FastAPI, File, UploadFile, HTTPException, Query
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import os
# from typing import Dict, Any, Optional
# import logging
# from dotenv import load_dotenv
# import base64
# from dotenv import load_dotenv

# load_dotenv()

# from image_processing import ImageProcessor

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI(title="Media Context Processor", version="2.0.0")
# app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# # Initialize processor
# image_processor = ImageProcessor(
#     serp_api_key=os.getenv("SERP_API_KEY"),
#     openai_api_key=os.getenv("OPENAI_API_KEY")
# )

# @app.post("/analyze-url/")
# async def analyze_image_url(image_url: str = Query(..., description="Image URL to analyze")) -> Dict[str, Any]:
#     """Analyze image from URL - main method"""
#     try:
#         result = image_processor.get_google_lens_context(image_url)
#         return {
#             "image_url": image_url,
#             "status": "success",
#             "lens_context": result
#         }
#     except Exception as e:
#         logger.error(f"Error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/health")
# async def health_check():
#     return {
#         "status": "healthy",
#         "features": {
#             "serp_api": bool(os.getenv("SERP_API_KEY")),
#             "openai": bool(os.getenv("OPENAI_API_KEY"))
#         }
#     }

# @app.get("/")
# async def root():
#     return {
#         "message": "Media Context Processor with Enhanced BOOM Fact-Check",
#         "endpoints": {
#             "analyze_url": "/analyze-url/",
#             "health": "/health"
#         },
#         "usage": "Use /analyze-url/ with image_url parameter to analyze images"
#     }

# if __name__ == "__main__":
#     required_vars = ["SERP_API_KEY"]
#     missing = [var for var in required_vars if not os.getenv(var)]
    
#     if missing:
#         logger.error(f"Missing: {missing}")
#         print("Required environment variables:")
#         print("  SERP_API_KEY (required)")
#         print("  OPENAI_API_KEY (optional, for better context)")
#         exit(1)
    
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from typing import Dict, Any, Optional
import logging
from dotenv import load_dotenv
import tempfile

load_dotenv()

from image_processing import ImageProcessor
from video_processing import VideoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Media Context Processor", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Initialize processors
image_processor = ImageProcessor(
    serp_api_key=os.getenv("SERP_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

video_processor = VideoProcessor(
    serp_api_key=os.getenv("SERP_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

@app.post("/analyze-url/")
async def analyze_image_url(image_url: str = Query(..., description="Image URL to analyze")) -> Dict[str, Any]:
    """Analyze image from URL"""
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

@app.post("/analyze-video/")
async def analyze_video_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Analyze uploaded video file"""
    try:
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            result = video_processor.get_video_context(tmp_file_path)
            return {
                "filename": file.filename,
                "status": "success",
                "video_context": result
            }
        finally:
            # Clean up temp file
            os.unlink(tmp_file_path)
            
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-video-url/")
async def analyze_video_url(video_url: str = Query(..., description="Video URL to analyze")) -> Dict[str, Any]:
    """Analyze video from URL"""
    try:
        # Download video to temp file
        import requests
        response = requests.get(video_url, stream=True, timeout=60)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_file_path = tmp_file.name
        
        try:
            result = video_processor.get_video_context(tmp_file_path, video_url=video_url)
            return {
                "video_url": video_url,
                "status": "success",
                "video_context": result
            }
        finally:
            os.unlink(tmp_file_path)
            
    except Exception as e:
        logger.error(f"Error processing video URL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "features": {
            "serp_api": bool(os.getenv("SERP_API_KEY")),
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "image_processing": True,
            "video_processing": True
        }
    }

@app.get("/")
async def root():
    return {
        "message": "Media Context Processor with Enhanced BOOM Fact-Check",
        "endpoints": {
            "analyze_url": "/analyze-url/",
            "analyze_video": "/analyze-video/",
            "analyze_video_url": "/analyze-video-url/",
            "health": "/health"
        },
        "usage": {
            "images": "Use /analyze-url/ with image_url parameter",
            "video_files": "Use /analyze-video/ with file upload",
            "video_urls": "Use /analyze-video-url/ with video_url parameter"
        }
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
