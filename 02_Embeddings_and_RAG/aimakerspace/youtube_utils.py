"""
YouTube utilities for video transcription and document loading
"""

import re
from typing import List, Dict, Any, Optional
from aimakerspace.text_utils import DocumentMetadata


class YouTubeTranscriber:
    """
    A mock YouTube transcriber that extracts video IDs and provides mock transcription.
    In a real scenario, this would use libraries like `yt-dlp` and `whisper`.
    """
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extracts YouTube video ID from various URL formats"""
        patterns = [
            r"(?:https?://)?(?:www\.)?(?:youtube\.com|youtu\.be)/(?:watch\?v=|embed/|v/|)([\w-]{11})(?:\S+)?",
            r"(?:https?://)?(?:www\.)?(?:youtube\.com|youtu\.be)/([\w-]{11})(?:\S+)?"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def get_video_info(self, url: str) -> Dict[str, Any]:
        """Mocks fetching video information"""
        video_id = self.extract_video_id(url)
        if not video_id:
            return {"error": "Invalid YouTube URL"}
        
        # Mock data
        return {
            "id": video_id,
            "title": f"Mock YouTube Video Title for {video_id}",
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "duration_seconds": 300,  # Mock duration
            "author": "Mock Author",
            "description": f"This is a mock description for video {video_id}.",
            "transcription_available": True
        }
    
    def transcribe_video(self, url: str) -> str:
        """Mocks transcribing a YouTube video"""
        video_info = self.get_video_info(url)
        if "error" in video_info:
            raise ValueError(video_info["error"])
        
        # Real transcription content for Power BI tutorial video
        return (
            f"Welcome to this Power BI tutorial for beginners. In this video, we'll cover the essential "
            f"steps to get started with Power BI Desktop. First, let's download and install Power BI Desktop "
            f"from the official Microsoft website. Once installed, we'll connect to our data sources. "
            f"I'll show you how to import data from Excel files and connect directly to SQL Server databases. "
            f"For our demonstration, we'll use a sales dataset containing customer information and transaction data. "
            f"We'll create several visualizations including bar charts to show sales by region, line graphs "
            f"for monthly trends, and pie charts for product category distribution. The tutorial covers "
            f"essential DAX formulas including SUM for calculating totals, AVERAGE for mean values, "
            f"and CALCULATE for conditional calculations. We'll also demonstrate advanced features like "
            f"creating relationships between the customers and transactions tables using primary and foreign keys. "
            f"The video shows how to use filters to drill down into specific time periods or regions. "
            f"We'll build an interactive dashboard with slicers that allow users to filter data dynamically. "
            f"Finally, we'll publish our report to the Power BI service and share it with team members "
            f"through organized workspaces. This enables collaborative data analysis and decision making."
        )


class YouTubeDocumentLoader:
    """Loads documents from YouTube video transcriptions"""
    
    def __init__(self, urls: List[str]):
        self.urls = urls
        self.documents = []
        self.metadata = []
        self.transcriber = YouTubeTranscriber()
    
    def load_documents(self) -> List[str]:
        """Load transcriptions from YouTube URLs"""
        for url in self.urls:
            try:
                video_info = self.transcriber.get_video_info(url)
                if "error" in video_info:
                    print(f"Warning: Could not get info for {url}: {video_info['error']}")
                    continue
                
                transcription = self.transcriber.transcribe_video(url)
                self.documents.append(transcription)
                
                # Create metadata for YouTube video
                metadata = DocumentMetadata(
                    filename=video_info['title'],
                    file_type="youtube",
                    size=len(transcription.encode('utf-8')),  # Approximate size
                    created_date=None,  # Not directly available from mock
                    modified_date=None,  # Not directly available from mock
                    page_count=1  # Treat as single document
                )
                metadata.set_content_hash(transcription)
                self.metadata.append(metadata)
                
            except Exception as e:
                print(f"Warning: Could not transcribe YouTube video {url}: {e}")
        
        return self.documents
