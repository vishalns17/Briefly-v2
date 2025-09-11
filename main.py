from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests
import feedparser
import json
import os
import time
import csv
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Dict, Any, Optional
import threading
import asyncio
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
app = FastAPI(title="RSS Feed Summarizer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","https://4bb48a35-d1a4-43d2-88cf-59c464af94f1-dev.e1-eu-north-azure.choreoapis.dev/default/briefly-backend/v1.0"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
max_text_length = 7000
feeds_file = "rss_feeds.json"
csv_file = "news_summaries.csv"
json_file = "news_summaries.json"

load_dotenv()
# Default settings
default = {
    'url': os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
    'api_key':os.environ.get("OPENAI_API_KEY",""),
    'model': os.environ.get("FEEDSUMMARIZER_MODEL", "gpt-3.5-turbo"),
    'system': os.environ.get("FEEDSUMMARIZER_SYSTEM", "You are an expert summarizer."),
    'instruction': os.environ.get("FEEDSUMMARIZER_INSTRUCTION", "Summarize this article into a short, punchy tech fact (max 2 sentences) to put in a newsletter, prioritizing the most important information first and then adding supporting details (inverted pyramid style). Categorize it into one of the following categories: AI, New in Tech, Business, Games/Entertainment. Return the response in the following JSON format only and do NOT include any markdown or escape characters inside it :{\"summary\": \"Your summary here\", \"tag\": \"Category\"}"),
    'maximum': int(os.environ.get("FEEDSUMMARIZER_MAX_ARTICLES", "100")),
    'dyk_prompt': os.environ.get("FEEDSUMMARIZER_DYK_INSTRUCTION","Turn this article into one fun, factual, and that feels like a surprising fact or hook for a newsletter. It should be exciting and attention-grabbing, but it does not have to start with 'Did you know'."),
    'time_lapse': int(os.environ.get("FEEDSUMMARIZER_TIME_LAPSE", "86400"))
}


# Pydantic models
class FeedRequest(BaseModel):
    url: str
    name: Optional[str] = ""

class FeedResponse(BaseModel):
    id: int
    name: str
    url: str

class ArticleResponse(BaseModel):
    id: int
    title: str
    url: str
    date: str
    author: str
    timestamp: str
    summary: str
    tag : str
    feed_name: Optional[str] = ""

class ArticleURLRequest(BaseModel):
    url: str

class NewsArticle:
    def __init__(self, entry, max_text_length):
        self.title = getattr(entry, 'title', 'Unknown')
        self.url = getattr(entry, 'link', 'NO LINK')
        self.date = getattr(entry, 'updated', getattr(entry, 'published', 'Unknown'))
        self.author = getattr(entry, 'author', 'Unknown')
        self.timestamp = datetime.now().isoformat()
        self.text = self.get_page_content(self.url, max_text_length)
        self.summary = ""
        self.feed_name = ""
        self.tag = ""
        
    def get_page_content(self, url, max_text_length):
        if url == "NO LINK":
            return "The feed entry doesn't seem to have any URL."
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except Exception as e:
            return f"The page {url} could not be loaded: {str(e)}"
        
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        
        if paragraphs:
            text = "\n".join(p.get_text() for p in paragraphs)
            words = text.split()
            if len(words) > max_text_length:
                text = " ".join(words[:max_text_length]) + "..."
            return f"Content of {url}:\n{text}"
        else:
            return f"The web page at {url} doesn't seem to have any readable content."
    
    def summarize(self, settings):
        if self.text and "doesn't seem to have any URL" not in self.text:
            self.summary, self.tag = generate_ai_response(self.text, settings)
        else:
            self.summary = "Could not summarize - no content available"
            self.tag = "Unknown"
        return self.summary
    
    def to_dict(self):
        return {
            'title': self.title,
            'url': self.url,
            'date': self.date,
            'author': self.author,
            'timestamp': self.timestamp,
            'summary': self.summary,
            'feed_name': getattr(self, 'feed_name', ''),
            'tag': self.tag
        }

def generate_ai_response(content, settings):
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings['api_key']}"
        }
        
        messages = [
            {"role": "system", "content": settings['system']},
            {"role": "user", "content": f"{content}\n\n{settings['instruction']}"}
        ]
        
        data = {
            'model': settings['model'],
            'messages': messages,
            'max_tokens': 600,
            'temperature': 0.7
        }
        
        # print(f"Sending request to {settings['url']}/chat/completions with data: {data}")
        
        response = requests.post(
            f"{settings['url']}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        print(f"Received response with status code: {response.status_code}")
        
        if response.status_code == 200:
            response_text = response.json()['choices'][0]['message']['content']
            # print(f"Response text: {response_text}")
            
            # Extract the JSON part from the response
            response_text = response_text.strip()
            start = response_text.find("{")
            end = response_text.rfind("}")
            if start != -1 and end != -1:
                response_text = response_text[start:end+1]
                # print(f"Extracted JSON string: {response_text}")
            else:
                print("No valid JSON found in response text")
                return f"Error parsing JSON response: {response_text}", "Unknown"
            
            # Removing any escape characters like \n or \t
            response_text = response_text.replace("\n", "").replace("\t", "")
            # print(f"Cleaned JSON string: {response_text}")
            
            try:
                response_json = json.loads(response_text)
                summary = response_json.get('summary', '').strip()
                tag = response_json.get('tag', 'Unknown').strip()
                # print(f"Parsed summary: {summary}, tag: {tag}")
                return summary, tag
            except json.JSONDecodeError:
                print(f"JSON decode error: {response_text}")
                return f"Error parsing JSON response: {response_text}", "Unknown"
        else:
            print(f"Error generating summary: {response.status_code}")
            return f"Error generating summary: {response.status_code}", "Unknown"
            
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return f"Error generating response: {str(e)}", "Unknown"

def fetch_article_text(url: str, max_text_length: int = 7000) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        return f"The page {url} could not be loaded: {str(e)}"

    soup = BeautifulSoup(response.content, "html.parser")
    paragraphs = soup.find_all("p")

    if paragraphs:
        text = "\n".join(p.get_text() for p in paragraphs)
        words = text.split()
        if len(words) > max_text_length:
            text = " ".join(words[:max_text_length]) + "..."
        return f"Content of {url}:\n{text}"
    else:
        return f"The web page at {url} doesn't seem to have any readable content."


def load_feeds():
    """Load RSS feeds from JSON file"""
    if os.path.exists(feeds_file):
        try:
            with open(feeds_file, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_feeds(feeds):
    """Save RSS feeds to JSON file"""
    with open(feeds_file, 'w') as f:
        json.dump(feeds, f, indent=2)

def save_to_csv(articles):
    """Save articles to CSV file"""
    fieldnames = ['title', 'url', 'date', 'author', 'timestamp', 'summary', 'feed_name','tag']
    
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for article in articles:
            writer.writerow(article.to_dict())

def clear_old_articles():
    """Remove articles older than a month from the JSON file"""
    if not os.path.exists(json_file):
        return
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
    except:
        return
    
    one_month_ago = datetime.now().replace(day=1) - pd.DateOffset(months=1)
    filtered_articles = []
    for article in articles:
        try:
            article_date = datetime.fromisoformat(article['timestamp'])
            if article_date >= one_month_ago:
                filtered_articles.append(article)
        except:
            # If timestamp is invalid, keep the article
            filtered_articles.append(article)
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_articles, f, indent=2, ensure_ascii=False)

def save_to_json(articles):
    """Append new articles to JSON file, avoiding duplicates by URL"""
    # Load existing articles
    existing_articles = []
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                existing_articles = json.load(f)
        except:
            existing_articles = []
    
    # Get set of existing URLs for quick lookup
    existing_urls = {art['url'] for art in existing_articles}
    
    # Filter new articles to avoid duplicates
    new_articles = [art for art in articles if art.to_dict()['url'] not in existing_urls]
    
    if new_articles:
        # Append new articles
        existing_articles.extend([art.to_dict() for art in new_articles])
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(existing_articles, f, indent=2, ensure_ascii=False)

def process_feeds_background():
    """Background process to handle RSS feeds - runs weekly"""
    print("Starting weekly RSS feed processing...")
    feeds = load_feeds()
    if not feeds:
        print("No feeds found to process")
        return
    
    settings = default.copy()
    all_articles = []
    
    for feed_info in feeds:
        try:
            print(f"Processing feed: {feed_info['name']}")
            feed = feedparser.parse(feed_info['url'])
            feed_title = getattr(feed.feed, 'title', feed_info['name'])
            
            articles = []
            now = time.time()
            
            for entry in feed.entries[:settings['maximum']]:
                if hasattr(entry, 'updated_parsed'):
                    then = time.mktime(entry.updated_parsed)
                elif hasattr(entry, 'published_parsed'):
                    then = time.mktime(entry.published_parsed)
                else:
                    then = now
                
                if (now - then) < settings['time_lapse']:
                    article = NewsArticle(entry, max_text_length)
                    article.feed_name = feed_title
                    articles.append(article)
            
            print(f"Found {len(articles)} recent articles from {feed_title}")
            
            for article in articles:
                article.summarize(settings)
                all_articles.append(article)
            
        except Exception as e:
            print(f"Error processing {feed_info['name']}: {str(e)}")
    
    if all_articles:
        clear_old_articles()  # Clear old articles before saving
        save_to_json(all_articles)
        print(f"Processed and saved {len(all_articles)} articles")
    else:
        print("No new articles to process")

# FastAPI Endpoints

@app.get("/", response_class=HTMLResponse)
async def get_home():
    """Serve the main HTML page"""
    return "Hi"

@app.get("/api/articles", response_model=List[ArticleResponse])
async def get_articles(limit: int = 100):
    """Get recent articles"""
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            
            articles.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            result = []
            for i, article in enumerate(articles[:limit], 1):
                result.append(ArticleResponse(
                    id=i,
                    title=article['title'],
                    url=article['url'],
                    date=article['date'],
                    author=article['author'],
                    timestamp=article['timestamp'],
                    summary=article['summary'],
                    feed_name=article.get('feed_name', ''),
                    tag=article['tag']
                ))
            
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading articles: {str(e)}")
    else:
        return []

@app.get("/api/article/{article_id}/summary")
async def get_article_summary(article_id: int):
    """Get summary for a specific article"""
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            
            articles.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            if 1 <= article_id <= len(articles):
                article = articles[article_id - 1]
                return {"summary": article['summary']}
            else:
                raise HTTPException(status_code=404, detail="Article not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading article: {str(e)}")
    else:
        raise HTTPException(status_code=404, detail="No articles available")

@app.get("/api/feeds", response_model=List[FeedResponse])
async def get_feeds():
    """Get all RSS feeds"""
    feeds = load_feeds()
    result = []
    for i, feed in enumerate(feeds, 1):
        result.append(FeedResponse(
            id=i,
            name=feed['name'],
            url=feed['url']
        ))
    return result

@app.post("/api/feeds")
async def add_feed(feed_request: FeedRequest):
    """Add a new RSS feed"""
    feeds = load_feeds()
    
    name = feed_request.name if feed_request.name else feed_request.url
    
    # Check if feed already exists
    for feed in feeds:
        if feed['url'] == feed_request.url:
            raise HTTPException(status_code=400, detail="Feed already exists")
    
    # Validate feed
    try:
        parsed_feed = feedparser.parse(feed_request.url)
        if parsed_feed.bozo:
            raise HTTPException(status_code=400, detail="Invalid RSS feed")
    except:
        raise HTTPException(status_code=400, detail="Cannot parse RSS feed")
    
    feeds.append({'url': feed_request.url, 'name': name})
    save_feeds(feeds)
    load_feeds()
    return {"message": f"Feed '{name}' added successfully"}

@app.delete("/api/feeds/{feed_id}")
async def remove_feed(feed_id: int):
    """Remove an RSS feed"""
    feeds = load_feeds()
    
    if 1 <= feed_id <= len(feeds):
        removed_feed = feeds.pop(feed_id - 1)
        save_feeds(feeds)
        load_feeds()
        return {"message": f"Feed '{removed_feed['name']}' removed successfully"}
    else:
        raise HTTPException(status_code=404, detail="Feed not found")

@app.post("/api/process-feeds")
async def manual_process_feeds(background_tasks: BackgroundTasks):
    """Manually trigger feed processing"""
    background_tasks.add_task(process_feeds_background)
    return {"message": "Feed processing started"}

@app.post("/api/convert-url")
async def convert_url_to_did_you_know(request: ArticleURLRequest):
    print("hi",request.url)
    article_text = fetch_article_text(request.url, max_text_length)

    if "could not be loaded" in article_text or "doesn't seem to have" in article_text:
        raise HTTPException(status_code=400, detail="Failed to extract readable content from the URL.")

    # Build instruction prompt
    settings = default.copy()
    prompt = f"{article_text}\n\n{settings['dyk_prompt']}"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {default['api_key']}"
    }

    messages = [
        {"role": "system", "content": default['system']},
        {"role": "user", "content": prompt}
    ]

    data = {
        "model": default["model"],
        "messages": messages,
        "max_tokens": 200,
        "temperature": 0.7,
    }

    try:
        response = requests.post(
            f"{default['url']}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()["choices"][0]["message"]["content"].strip()
            return {"did_you_know": result}
        else:
            raise HTTPException(status_code=500, detail="Failed to generate summary from LLM.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during LLM call: {str(e)}")

# Initialize scheduler for weekly processing
scheduler = BackgroundScheduler()

@app.on_event("startup")
async def startup_event():
    # Create initial feeds file with sample feed if it doesn't exist
    if not os.path.exists(feeds_file):
        sample_feeds = [
            {"url": "https://news.ycombinator.com/rss", "name": "Hacker News"},
            {"url": "https://feeds.bbci.co.uk/news/rss.xml", "name": "BBC News"}
        ]
        save_feeds(sample_feeds)
    
    # Schedule weekly processing (every Monday at 9:00 AM)
    scheduler.add_job(
        process_feeds_background,
        CronTrigger(day_of_week=0, hour=9, minute=0),  # 0 = Monday
        id='weekly_feed_processing'
    )
    scheduler.start()
    print("RSS Feed Summarizer started with weekly processing enabled")

@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown()

if __name__ == "__main__":
    print("Starting RSS Feed Summarizer Backend...")
    print("FastAPI will run on http://localhost:8000")
    print("Automatic weekly processing is enabled (Mondays at 9:00 AM)")
    process_feeds_background()
    print("Commencing automatic processing")
    uvicorn.run(app, host="0.0.0.0", port=8000)