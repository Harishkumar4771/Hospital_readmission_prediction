import json
import sys
import os
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

try:
    from app import app as flask_app
    app_loaded = True
except Exception as e:
    app_loaded = False
    error_msg = str(e)

def handler(event, context):
    """Handle Netlify function requests"""
    if not app_loaded:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': f'Failed to load app: {error_msg}'}),
            'headers': {'Content-Type': 'application/json'}
        }
    
    try:
        http_method = event.get('httpMethod', 'GET').upper()
        path = event.get('path', '/')
        headers = event.get('headers', {})
        body = event.get('body', '')
        
        # Create test client and make request
        with flask_app.test_client() as client:
            if http_method == 'GET':
                response = client.get(path, headers=headers)
            elif http_method == 'POST':
                response = client.post(
                    path, 
                    data=body,
                    headers=headers,
                    content_type=headers.get('content-type', 'application/json')
                )
            else:
                response = client.open(path, method=http_method, data=body, headers=headers)
            
            return {
                'statusCode': response.status_code,
                'body': response.get_data(as_text=True),
                'headers': dict(response.headers)
            }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)}),
            'headers': {'Content-Type': 'application/json'}
        }
