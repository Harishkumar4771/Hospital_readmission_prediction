import os
import sys

# Add parent directory to path to import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

def handler(event, context):
    """Netlify serverless function handler for Flask app"""
    return {
        "statusCode": 200,
        "body": "Hello from Netlify Functions"
    }
