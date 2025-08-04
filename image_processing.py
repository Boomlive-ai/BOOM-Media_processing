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
        """Use GPT-4.1-mini to extract common context from search results"""
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