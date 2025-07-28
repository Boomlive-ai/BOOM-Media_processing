import cv2
import asyncio
import base64
import io
from PIL import Image
from typing import Dict, Any, List, Optional
import logging
import numpy as np
from image_processing import ImageProcessor

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize VideoProcessor
        
        Args:
            api_key: OpenAI API key (passed to ImageProcessor)
        """
        self.image_processor = ImageProcessor(api_key)
    
    def extract_frames(self, video_path: str, frame_interval: int = 30) -> List[np.ndarray]:
        """
        Extract frames from video at specified intervals
        
        Args:
            video_path: Path to video file
            frame_interval: Extract every Nth frame
            
        Returns:
            List of frame arrays
        """
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Video info: {total_frames} frames, {fps} FPS")
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Extract frame at specified intervals
                if frame_count % frame_interval == 0:
                    frames.append(frame)
                    logger.info(f"Extracted frame {frame_count} ({len(frames)} total)")
                
                frame_count += 1
                
                # Limit to reasonable number of frames to avoid excessive API calls
                if len(frames) >= 20:  # Max 20 frames
                    logger.info("Reached maximum frame limit (20)")
                    break
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            raise
        
        return frames
    
    def frame_to_bytes(self, frame: np.ndarray) -> bytes:
        """
        Convert frame array to bytes
        
        Args:
            frame: OpenCV frame array
            
        Returns:
            Image bytes
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Save to bytes
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format='PNG')
        
        return img_bytes.getvalue()
    
    async def process_frame(
        self, 
        frame: np.ndarray, 
        frame_number: int,
        extract_text: bool = True,
        get_context: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single frame
        
        Args:
            frame: Frame array
            frame_number: Frame number for reference
            extract_text: Whether to extract text
            get_context: Whether to get context
            
        Returns:
            Frame processing results
        """
        try:
            # Convert frame to bytes
            frame_bytes = self.frame_to_bytes(frame)
            
            # Process using image processor
            result = await self.image_processor.process_image(
                frame_bytes,
                extract_text=extract_text,
                get_context=get_context
            )
            
            # Add frame metadata
            result['frame_number'] = frame_number
            result['timestamp'] = f"{frame_number / 30:.2f}s"  # Approximate timestamp
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_number}: {e}")
            return {
                'frame_number': frame_number,
                'error': str(e)
            }
    
    async def process_video(
        self,
        video_path: str,
        frame_interval: int = 30,
        extract_text: bool = True,
        get_context: bool = True,
        max_concurrent: int = 3
    ) -> Dict[str, Any]:
        """
        Process video to extract text and/or context from frames
        
        Args:
            video_path: Path to video file
            frame_interval: Extract every Nth frame
            extract_text: Whether to extract text from frames
            get_context: Whether to get context description
            max_concurrent: Maximum concurrent API calls
            
        Returns:
            Video processing results
        """
        try:
            logger.info(f"Starting video processing: {video_path}")
            
            # Extract frames
            frames = self.extract_frames(video_path, frame_interval)
            
            if not frames:
                return {"error": "No frames could be extracted from video"}
            
            logger.info(f"Processing {len(frames)} frames")
            
            # Create semaphore to limit concurrent API calls
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_frame_with_semaphore(frame, frame_num):
                async with semaphore:
                    return await self.process_frame(
                        frame, 
                        frame_num * frame_interval,
                        extract_text=extract_text,
                        get_context=get_context
                    )
            
            # Process frames concurrently
            tasks = [
                process_frame_with_semaphore(frame, i) 
                for i, frame in enumerate(frames)
            ]
            
            frame_results = await asyncio.gather(*tasks)
            
            # Compile results
            result = {
                "total_frames_processed": len(frames),
                "frame_interval": frame_interval,
                "frames": frame_results
            }
            
            # Generate summary
            if extract_text:
                all_text = []
                for frame_result in frame_results:
                    if 'text' in frame_result and frame_result['text'] != 'No text found':
                        all_text.append(frame_result['text'])
                
                result['extracted_text_summary'] = {
                    "total_frames_with_text": len(all_text),
                    "combined_text": " ".join(all_text) if all_text else "No text found in video"
                }
            
            if get_context:
                contexts = [
                    frame_result.get('context', '') 
                    for frame_result in frame_results 
                    if 'context' in frame_result
                ]
                
                result['context_summary'] = {
                    "total_frames_analyzed": len(contexts),
                    "sample_descriptions": contexts[:3]  # First 3 descriptions as sample
                }
            
            logger.info("Video processing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return {"error": str(e)}