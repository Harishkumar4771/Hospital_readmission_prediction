import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app

# This is the entry point for Vercel serverless functions
def handler(request):
    """Vercel serverless function handler for Flask app"""
    return app(request.environ, request.start_response)
