# import openai
# import base64
# import io
# from PIL import Image
# from typing import Dict, Any, Optional
# import asyncio
# import logging

# logger = logging.getLogger(__name__)

# class ImageProcessor:
#     def __init__(self, api_key: Optional[str] = None):
#         """
#         Initialize ImageProcessor with OpenAI API key
        
#         Args:
#             api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
#         """
#         if api_key:
#             openai.api_key = api_key
#         else:
#             # Will use OPENAI_API_KEY environment variable
#             pass
        
#         self.client = openai.AsyncOpenAI()
    
#     def encode_image(self, image_content: bytes) -> str:
#         """
#         Encode image bytes to base64 string
        
#         Args:
#             image_content: Raw image bytes
            
#         Returns:
#             Base64 encoded image string
#         """
#         return base64.b64encode(image_content).decode('utf-8')
    
#     def resize_image_if_needed(self, image_content: bytes, max_size: int = 1024) -> bytes:
#         """
#         Resize image if it's too large to optimize API calls
        
#         Args:
#             image_content: Raw image bytes
#             max_size: Maximum dimension in pixels
            
#         Returns:
#             Resized image bytes
#         """
#         try:
#             img = Image.open(io.BytesIO(image_content))
            
#             # Check if resizing is needed
#             if max(img.width, img.height) > max_size:
#                 # Calculate new dimensions maintaining aspect ratio
#                 ratio = max_size / max(img.width, img.height)
#                 new_width = int(img.width * ratio)
#                 new_height = int(img.height * ratio)
                
#                 # Resize image
#                 img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
#                 # Save to bytes
#                 output = io.BytesIO()
#                 img.save(output, format='PNG')
#                 return output.getvalue()
            
#             return image_content
            
#         except Exception as e:
#             logger.warning(f"Could not resize image: {e}")
#             return image_content
    
#     async def extract_text_from_image(self, image_base64: str) -> str:
#         """
#         Extract text from image using GPT-4 Vision
        
#         Args:
#             image_base64: Base64 encoded image
            
#         Returns:
#             Extracted text
#         """
#         try:
#             response = await self.client.chat.completions.create(
#                 model="gpt-4.1-mini",  # Use GPT-4o for vision tasks
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": [
#                             {
#                                 "type": "text",
#                                 "text": "Describe the image and identify any public figures or locations, fetch any claim in the image. Return results in JSON.Extract all text from this image. Return only the text content, no additional commentary. If there's no text, respond with 'No text found'."
#                             },
#                             {
#                                 "type": "image_url",
#                                 "image_url": {
#                                     "url": f"data:image/png;base64,{image_base64}",
#                                     "detail": "high"
#                                 }
#                             }
#                         ]
#                     }
#                 ],
#                 max_tokens=1000,
#                 temperature=0.1
#             )
            
#             return response.choices[0].message.content.strip()
            
#         except Exception as e:
#             logger.error(f"Error extracting text: {e}")
#             return f"Error extracting text: {str(e)}"
    
#     async def get_image_context(self, image_base64: str) -> str:
#         """
#         Get contextual description of image using GPT-4 Vision
        
#         Args:
#             image_base64: Base64 encoded image
            
#         Returns:
#             Context description
#         """
#         try:
#             response = await self.client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": [
#                             {
#                                 "type": "text",
#                                 "text": """Analyze this image and provide a comprehensive description including:
# 1. Scene/setting description
# 2. Objects and people present
# 3. Activities or actions taking place
# 4. Mood or atmosphere
# 5. Any notable details or context clues

# Be detailed but concise."""
#                             },
#                             {
#                                 "type": "image_url",
#                                 "image_url": {
#                                     "url": f"data:image/png;base64,{image_base64}",
#                                     "detail": "high"
#                                 }
#                             }
#                         ]
#                     }
#                 ],
#                 max_tokens=1500,
#                 temperature=0.3
#             )
            
#             return response.choices[0].message.content.strip()
            
#         except Exception as e:
#             logger.error(f"Error getting context: {e}")
#             return f"Error getting context: {str(e)}"
    
#     async def process_image(
#         self, 
#         image_content: bytes, 
#         extract_text: bool = True, 
#         get_context: bool = True
#     ) -> Dict[str, Any]:
#         """
#         Process image to extract text and/or context
        
#         Args:
#             image_content: Raw image bytes
#             extract_text: Whether to extract text
#             get_context: Whether to get context description
            
#         Returns:
#             Dictionary with processing results
#         """
#         try:
#             # Resize image if needed to optimize API calls
#             processed_content = self.resize_image_if_needed(image_content)
            
#             # Encode to base64
#             image_base64 = self.encode_image(processed_content)
            
#             result = {}
            
#             # Create tasks for parallel processing
#             tasks = []
            
#             if extract_text:
#                 tasks.append(("text", self.extract_text_from_image(image_base64)))
            
#             if get_context:
#                 tasks.append(("context", self.get_image_context(image_base64)))
            
#             # Execute tasks concurrently
#             if tasks:
#                 task_results = await asyncio.gather(*[task[1] for task in tasks])
                
#                 for i, (task_name, _) in enumerate(tasks):
#                     result[task_name] = task_results[i]
            
#             return result
            
#         except Exception as e:
#             logger.error(f"Error processing image: {e}")
#             return {"error": str(e)}
import openai
import requests
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self, serp_api_key: str, openai_api_key: Optional[str] = None):
        self.serp_api_key = serp_api_key
        if openai_api_key:
            self.client = openai.OpenAI(api_key=openai_api_key)
    
    def get_google_lens_context(self, image_url: str) -> Dict[str, Any]:
        """Get Google Lens context with BOOM fact-check filtering using requests"""
        try:
            # Direct API request to SerpAPI
            url = "https://serpapi.com/search"
            params = {
                "engine": "google_lens",
                "hl": "en",
                "country": "in",
                "url": image_url,
                "api_key": self.serp_api_key
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            results = response.json()
            
            visual_matches = results.get("visual_matches", [])
            
            # Filter BOOM fact-check articles
            boom_articles = [
                match for match in visual_matches 
                if "boomlive.in" in match.get("link", "").lower()
            ]
            
            if boom_articles:
                return {
                    "source": "boom_factcheck",
                    "context": self._generate_boom_fact_check_query(boom_articles),
                    "articles": boom_articles[:3]  # Top 3 BOOM articles
                }
            
            # If no BOOM articles, get general context using LLM
            return {
                "source": "general_context", 
                "context": self._get_general_context(visual_matches),
                "matches": visual_matches[:5]  # Top 5 matches
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"SerpAPI request error: {e}")
            return {"error": f"API request failed: {str(e)}"}
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {"error": str(e)}
    
    def _extract_boom_context(self, boom_articles: List[Dict]) -> str:
        """Extract context directly from BOOM articles (fallback method)"""
        contexts = []
        for article in boom_articles:
            title = article.get("title", "")
            snippet = article.get("snippet", "")
            link = article.get("link", "")
            contexts.append(f"{title}. {snippet}. {link}")
        return " ".join(contexts)
    
    def _generate_boom_fact_check_query(self, boom_articles: List[Dict]) -> str:
        """Generate fact-check claim query using LLM with BOOM articles"""
        if not hasattr(self, 'client') or not boom_articles:
            return self._extract_boom_context(boom_articles)
        
        try:
            # Prepare BOOM articles for LLM
            boom_text = "\n".join([
                f"Title: {article.get('title', 'N/A')}\nSnippet: {article.get('snippet', 'N/A')}\nLink: {article.get('link', 'N/A')}\n"
                for article in boom_articles
            ])
            
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{
                    "role": "user", 
                    "content": f"""Based on these BOOM fact-check articles, generate a clear fact-check claim query with the link:

{boom_text}



Format as a clear, informative claim query fetched context from boom fct check articles."""
                }],
                max_tokens=300,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating BOOM fact-check query: {e}")
            return self._extract_boom_context(boom_articles)
    
    def _get_general_context(self, visual_matches: List[Dict]) -> str:
        """Use GPT-4o-mini to extract common context from search results"""
        if not hasattr(self, 'client') or not visual_matches:
            return "No context available"
        
        try:
            # Prepare search results for LLM
            results_text = "\n".join([
                f"Title: {match.get('title', 'N/A')}\nSnippet: {match.get('snippet', 'N/A')}\n"
                for match in visual_matches[:10]
            ])
            
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{
                    "role": "user", 
                    "content": f"""Analyze these search results and provide a concise context claim query about what this image likely shows:

{results_text}

Focus on the most common themes and factual information for generating the claim query."""
                }],
                max_tokens=200,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error getting LLM context: {e}")
            return "Context extraction failed"

# Usage:
# processor = ImageProcessor(serp_api_key="your_key", openai_api_key="your_openai_key")
# result = processor.get_google_lens_context("https://image-url.jpg")