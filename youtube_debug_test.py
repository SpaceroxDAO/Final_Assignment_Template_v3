#!/usr/bin/env python3
"""
YouTube Debug Test Suite
Focused on debugging and fixing the YouTube title extraction issue
"""

import os
import sys
import logging
import re

# Setup detailed logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Import the agent code
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import BasicAgent

def test_youtube_url_parsing():
    """Test YouTube URL parsing and video ID extraction"""
    test_cases = [
        {
            'url': 'https://www.youtube.com/watch?v=jNQXAC9IVRw',
            'expected_id': 'jNQXAC9IVRw'
        },
        {
            'url': 'https://www.youtube.com/watch?v=jNQXAC9IVRw?',
            'expected_id': 'jNQXAC9IVRw'
        },
        {
            'url': 'https://youtu.be/jNQXAC9IVRw',
            'expected_id': 'jNQXAC9IVRw'
        },
        {
            'url': 'https://www.youtube.com/watch?v=jNQXAC9IVRw&feature=youtu.be',
            'expected_id': 'jNQXAC9IVRw'
        }
    ]
    
    for test in test_cases:
        url = test['url']
        expected = test['expected_id']
        
        # Test the parsing logic
        video_id = None
        if 'youtube.com/watch?v=' in url:
            video_id = url.split('v=')[1].split('&')[0].split('?')[0]
        elif 'youtu.be/' in url:
            video_id = url.split('youtu.be/')[1].split('?')[0]
        
        print(f"URL: {url}")
        print(f"Extracted ID: {video_id}")
        print(f"Expected ID: {expected}")
        print(f"Match: {video_id == expected}")
        print("-" * 50)

def debug_youtube_extraction():
    """Debug the YouTube title extraction process"""
    agent = BasicAgent()
    
    # Test question
    question = 'What is the title of the YouTube video at https://www.youtube.com/watch?v=jNQXAC9IVRw?'
    
    print("Testing YouTube title extraction...")
    print(f"Question: {question}")
    
    # Debug the parsing
    print("\n--- Testing URL Extraction ---")
    youtube_url_pattern = r'https?://(?:www\.)?youtube\.com/watch\?v=[^\s]+|https?://youtu\.be/[^\s]+'
    youtube_match = re.search(youtube_url_pattern, question)
    if youtube_match:
        url = youtube_match.group(0)
        print(f"Extracted URL: {url}")
        
        # Clean trailing ?
        if url.endswith('?'):
            url = url[:-1]
            print(f"Cleaned URL: {url}")
        
        # Extract video ID
        video_id = None
        if 'youtube.com/watch?v=' in url:
            video_id = url.split('v=')[1].split('&')[0].split('?')[0]
        elif 'youtu.be/' in url:
            video_id = url.split('youtu.be/')[1].split('?')[0]
        
        print(f"Video ID: {video_id}")
    
    # Run the agent
    print("\n--- Running Agent ---")
    try:
        result = agent(question)
        print(f"Agent Result: '{result}'")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def test_fetch_youtube_data_directly():
    """Test the _fetch_youtube_data method directly"""
    agent = BasicAgent()
    
    if not agent.youtube_service:
        print("YouTube service not initialized - skipping direct API test")
        return
    
    print("\n--- Testing _fetch_youtube_data directly ---")
    
    # Test with clean video ID
    video_id = 'jNQXAC9IVRw'
    print(f"Testing with video ID: {video_id}")
    
    try:
        # Simulate the _fetch_youtube_data call
        response = agent.youtube_service.videos().list(
            part="snippet",
            id=video_id
        ).execute()
        
        if 'items' in response and response['items']:
            video_data = response['items'][0]['snippet']
            title = video_data.get('title', '')
            print(f"API Response Title: '{title}'")
            
            # Test the full context text generation
            description = video_data.get('description', '')[:200]
            tags = video_data.get('tags', [])[:5]
            channel = video_data.get('channelTitle', '')
            
            context_text = f"YouTube Video Information:\n"
            context_text += f"Title: {title}\n"
            context_text += f"Channel: {channel}\n"
            context_text += f"Description: {description}\n"
            if tags:
                context_text += f"Tags: {', '.join(tags)}\n"
            
            print("\nGenerated context text:")
            print(context_text)
    except Exception as e:
        print(f"API Error: {e}")
        import traceback
        traceback.print_exc()

def test_fallback_extraction():
    """Test the fallback HTML extraction"""
    print("\n--- Testing Fallback HTML Extraction ---")
    
    # Simulate what happens when API fails
    import requests
    from bs4 import BeautifulSoup
    
    url = 'https://www.youtube.com/watch?v=jNQXAC9IVRw'
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try different selectors for title
            selectors = [
                'meta[property="og:title"]',
                'meta[name="title"]',
                'title'
            ]
            
            for selector in selectors:
                element = soup.select_one(selector)
                if element:
                    if selector.startswith('meta'):
                        content = element.get('content', '')
                    else:
                        content = element.get_text(strip=True)
                    
                    print(f"Selector: {selector}")
                    print(f"Content: '{content}'")
                    
                    # Clean YouTube suffix
                    if content and ' - YouTube' in content:
                        content = content.replace(' - YouTube', '').strip()
                        print(f"Cleaned: '{content}'")
                    print("-" * 30)
    except Exception as e:
        print(f"Request Error: {e}")

def main():
    """Run all debug tests"""
    print("Starting YouTube Debug Tests\n")
    
    print("=== Test 1: URL Parsing ===")
    test_youtube_url_parsing()
    
    print("\n=== Test 2: Full Extraction Debug ===")
    debug_youtube_extraction()
    
    print("\n=== Test 3: Direct API Test ===")
    test_fetch_youtube_data_directly()
    
    print("\n=== Test 4: Fallback Extraction ===")
    test_fallback_extraction()

if __name__ == "__main__":
    main()