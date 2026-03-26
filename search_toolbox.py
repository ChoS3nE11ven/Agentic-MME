# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List


def build_search_tools_schema() -> List[Dict[str, Any]]:
    """OpenAI tools schema for web + visual search.

    Notes for reviewers:
    - Web search uses Serper.dev (Google Search API)
    - Visual search uses Serper.dev Lens
    - If Lens is called on a local image path/data URL, we first upload to an
      external image host (default: ImgBB) to obtain a public image URL.
      (No local ToolHub / FastAPI server.)
    """

    return [
        {
            "type": "function",
            "function": {
                "name": "google_search",
                "description": "Search the web using Google via Serper.dev API. Use for facts, current information, specifications, prices, or any knowledge queries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query."},
                        "gl": {"type": "string", "description": "Geo location code (e.g., 'us', 'cn'). Default: 'us'"},
                        "hl": {"type": "string", "description": "Language code (e.g., 'en', 'zh'). Default: 'en'"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "google_lens_search",
                "description": "Reverse image search using Google Lens via Serper.dev API. Use to identify objects, brands, logos, landmarks, products, or text in images.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_ref": {
                            "type": "string",
                            "description": "Quick reference: 'current' for the latest processed image, 'original' for the input image.",
                            "enum": ["current", "original"],
                        },
                        "image_path": {
                            "type": "string",
                            "description": "Filename or full path to a specific image. After image processing, you can use the filename (e.g., 'transformed_image_0.png') to search that specific image.",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_webpage",
                "description": "Fetch and read the content of a webpage. Returns clean text extracted from the URL via Jina Reader.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The webpage URL to fetch (must be http/https)."},
                        "max_chars": {"type": "integer", "description": "Maximum characters to return. Default: 12000"},
                    },
                    "required": ["url"],
                },
            },
        },
    ]
