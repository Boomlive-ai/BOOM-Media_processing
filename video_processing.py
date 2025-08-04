import openai
import requests
import cv2
import os
from typing import Dict, Any, Optional, List
import logging
import numpy as np
from utils import upload_frame_to_imgbb  # Make sure utils.py is in the same folder or in PYTHONPATH

logger = logging.getLogger(__name__)

TRUSTED_SOURCES = [
    "bbc.com/hindi",
    "bbc.com/marathi",
    "bbc.com/news/world/asia/india",
    "indianexpress.com",
    "thenewsminute.com",
    "thehindu.com",
    "indiaspendhindi.com",
    "indiaspend.com"
]

class VideoProcessor:
    def __init__(self, serp_api_key: str, openai_api_key: Optional[str] = None):
        self.serp_api_key = serp_api_key
        if openai_api_key:
            self.client = openai.OpenAI(api_key=openai_api_key)
        else:
            self.client = None

    def extract_key_frames(self, video_path: str, num_frames: int = 5) -> List[np.ndarray]:
        """Extract key frames from video for analysis"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_intervals = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            frames = []
            for frame_num in frame_intervals:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()
            return frames
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return []

    def get_video_context(self, video_path: str, video_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze video, returning full context, a concise query, and trusted-source article links.
        """
        try:
            frames = self.extract_key_frames(video_path)
            if not frames:
                return {"error": "Could not extract frames from video"}

            all_results = []
            boom_articles = []

            for i, frame in enumerate(frames):
                img_url = upload_frame_to_imgbb(frame)
                if img_url:
                    frame_result = self.get_google_lens_context_for_frame(img_url, frame_index=i)
                    all_results.append(frame_result)
                    if frame_result.get("source") == "boom_factcheck":
                        boom_articles.extend(frame_result.get("articles", []))

            results_obj = {
                "frame_results": all_results,
                "frame_count": len(frames),
                "video_info": {
                    "source": video_url or video_path,
                    "analyzed_frames": len(frames)
                }
            }

            if boom_articles:
                results_obj["source"] = "boom_factcheck"
                results_obj["context"] = self._generate_video_boom_fact_check_query(boom_articles)
                results_obj["articles"] = boom_articles[:5]
            else:
                results_obj["source"] = "general_context"
                results_obj["context"] = self._get_video_general_context(all_results)

            # Add concise query & trusted links
            sorter_query, trusted_links = self.get_sorter_query_and_trusted_links(all_results)
            results_obj["concise_query"] = sorter_query
            results_obj["trusted_links"] = trusted_links

            return results_obj

        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return {"error": str(e)}

    def get_google_lens_context_for_frame(self, frame_url: str, frame_index: int = 0) -> Dict[str, Any]:
        """Get Google Lens context for a single frame"""
        try:
            url = "https://serpapi.com/search"
            params = {
                "engine": "google_lens",
                "hl": "en",
                "country": "in",
                "url": frame_url,
                "api_key": self.serp_api_key
            }
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            results = response.json()
            visual_matches = results.get("visual_matches", [])
            boom_articles = [
                match for match in visual_matches 
                if "boomlive.in" in match.get("link", "").lower()
            ]
            if boom_articles:
                return {
                    "source": "boom_factcheck",
                    "context": self._generate_boom_fact_check_query(boom_articles),
                    "articles": boom_articles[:3],
                    "frame_index": frame_index
                }
            return {
                "source": "general_context",
                "context": self._get_general_context(visual_matches),
                "matches": visual_matches[:5],
                "frame_index": frame_index
            }
        except Exception as e:
            logger.error(f"Error processing frame {frame_index}: {e}")
            return {"error": str(e), "frame_index": frame_index}

    def get_sorter_query_and_trusted_links(self, frame_results: List[Dict]) -> (str, List[str]):
        """
        Generate a concise search query (from frame contexts and matches) and extract trusted article links.
        """
        # Aggregate keywords from all frame contexts and titles
        keywords = []
        trusted_links = set()
        for fr in frame_results:
            if "context" in fr:
                # Tokenize and keep words (you may want to do more advanced NLP)
                words = fr["context"].lower().split()
                keywords += [w.strip(",.#:;!()") for w in words if len(w) > 3]
            for match in fr.get("matches", []):
                for trusted in TRUSTED_SOURCES:
                    if trusted in match.get("link", ""):
                        trusted_links.add(match["link"])
                # Add titles to keywords as well
                title_words = match.get("title", "").lower().split()
                keywords += [w.strip(",.#:;!()") for w in title_words if len(w) > 3]
        # Remove duplicates, limit length
        keywords = list(dict.fromkeys(keywords))
        sorter_query = " ".join(keywords[:30])  # Limit to about 30 main keywords
        return sorter_query, list(trusted_links)

    # ---- Everything below is unchanged from your code ----

    def search_video_by_description(self, description: str) -> Dict[str, Any]:
        try:
            url = "https://serpapi.com/search"
            params = {
                "engine": "google",
                "q": f"{description} site:boomlive.in OR fact check video",
                "tbm": "vid",
                "api_key": self.serp_api_key
            }
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            results = response.json()
            video_results = results.get("video_results", [])
            boom_videos = [
                video for video in video_results
                if "boomlive.in" in video.get("link", "").lower()
            ]
            return {
                "source": "video_search",
                "boom_videos": boom_videos,
                "all_videos": video_results[:10]
            }
        except Exception as e:
            logger.error(f"Error in video search: {e}")
            return {"error": str(e)}

    def _generate_video_boom_fact_check_query(self, boom_articles: List[Dict]) -> str:
        if not self.client or not boom_articles:
            return self._extract_boom_context(boom_articles)
        try:
            unique_articles = []
            seen_links = set()
            for article in boom_articles:
                link = article.get('link', '')
                if link not in seen_links:
                    unique_articles.append(article)
                    seen_links.add(link)
            boom_text = "\n".join([
                f"Title: {article.get('title', 'N/A')}\nSnippet: {article.get('snippet', 'N/A')}\nLink: {article.get('link', 'N/A')}\n"
                for article in unique_articles
            ])
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{
                    "role": "user", 
                    "content": f"""Based on these BOOM fact-check articles found across multiple video frames, generate a comprehensive video fact-check claim query:\n\n{boom_text}\n\nSince this came from video analysis, focus on:\n1. The main claim or narrative in the video\n2. Any visual misinformation or manipulation\n3. Context about what the video actually shows vs. what it claims\n4. Include relevant BOOM fact-check links\n\nFormat as a clear, informative video fact-check summary."""
                }],
                max_tokens=400,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating video BOOM fact-check query: {e}")
            return self._extract_boom_context(boom_articles)

    def _get_video_general_context(self, frame_results: List[Dict]) -> str:
        if not self.client or not frame_results:
            return "No context available"
        try:
            contexts = []
            for result in frame_results:
                if result.get("context"):
                    contexts.append(f"Frame {result.get('frame_index', 'N/A')}: {result['context']}")
            if not contexts:
                return "No context extracted from video frames"
            combined_context = "\n".join(contexts)
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{
                    "role": "user", 
                    "content": f"""Analyze these contexts extracted from different frames of a video and provide a comprehensive video analysis:\n\n{combined_context}\n\nProvide:\n1. Main subject/content of the video\n2. Common themes across frames  \n3. Any potential misinformation patterns\n4. Overall video context and claim\n\nFormat as a coherent video analysis summary."""
                }],
                max_tokens=300,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error getting video LLM context: {e}")
            return "Video context extraction failed"

    def _extract_boom_context(self, boom_articles: List[Dict]) -> str:
        contexts = []
        for article in boom_articles:
            title = article.get("title", "")
            snippet = article.get("snippet", "")
            link = article.get("link", "")
            contexts.append(f"{title}. {snippet}. {link}")
        return " ".join(contexts)

    def _generate_boom_fact_check_query(self, boom_articles: List[Dict]) -> str:
        if not self.client or not boom_articles:
            return self._extract_boom_context(boom_articles)
        try:
            boom_text = "\n".join([
                f"Title: {article.get('title', 'N/A')}\nSnippet: {article.get('snippet', 'N/A')}\nLink: {article.get('link', 'N/A')}\n"
                for article in boom_articles
            ])
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{
                    "role": "user", 
                    "content": f"""Based on these BOOM fact-check articles, generate a clear fact-check claim query:\n\n{boom_text}\n\nFormat as a clear, informative claim query with context from BOOM fact-check articles."""
                }],
                max_tokens=300,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating BOOM fact-check query: {e}")
            return self._extract_boom_context(boom_articles)

    def _get_general_context(self, visual_matches: List[Dict]) -> str:
        if not self.client or not visual_matches:
            return "No context available"
        try:
            results_text = "\n".join([
                f"Title: {match.get('title', 'N/A')}\nSnippet: {match.get('snippet', 'N/A')}\n"
                for match in visual_matches[:10]
            ])
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{
                    "role": "user", 
                    "content": f"""Analyze these search results from a video frame and provide context query :\n\n{results_text}\n\nFocus on identifying what this frame likely shows and any factual information."""
                }],
                max_tokens=200,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error getting LLM context: {e}")
            return "Context extraction failed"
