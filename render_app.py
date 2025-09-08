# render_app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from datetime import datetime, timedelta
import os

app = FastAPI(title="Guardian AI Render Service")

# ==============================
# CONFIG
# ==============================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX", "YOUR_GOOGLE_CX_ID")  # Custom Search Engine ID
SERP_API_KEY = os.getenv("SERP_API_KEY", "YOUR_SERP_API_KEY")

GOOGLE_LIMIT = 100  # daily free quota
SERP_LIMIT = 3      # daily fallback quota

# Usage trackers
usage = {"google": 0, "serp": 0}
reset_time = datetime.now() + timedelta(days=1)

# Lazy load summarizer to save memory
summarizer = None
def get_summarizer():
    global summarizer
    if summarizer is None:
        summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")
    return summarizer

class QuestionRequest(BaseModel):
    question: str

# ==============================
# HELPERS
# ==============================
def reset_usage_if_needed():
    global reset_time, usage
    if datetime.now() >= reset_time:
        usage = {"google": 0, "serp": 0}
        reset_time = datetime.now() + timedelta(days=1)


def search_web(query, num_results=3):
    """Try Google first, fallback to SerpAPI, then static URLs."""
    reset_usage_if_needed()
    urls = []

    # 1. Google Programmable Search API
    if usage["google"] < GOOGLE_LIMIT:
        try:
            params = {"q": query, "key": GOOGLE_API_KEY, "cx": GOOGLE_CX, "num": num_results}
            r = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=5)
            results = r.json().get("items", [])
            urls = [item["link"] for item in results if "link" in item]
            if urls:
                usage["google"] += 1
                return urls[:num_results]
        except Exception:
            pass

    # 2. SerpAPI fallback
    if usage["serp"] < SERP_LIMIT:
        try:
            params = {"q": query, "api_key": SERP_API_KEY, "num": num_results}
            r = requests.get("https://serpapi.com/search", params=params, timeout=5)
            results = r.json().get("organic_results", [])
            urls = [r["link"] for r in results if "link" in r]
            if urls:
                usage["serp"] += 1
                return urls[:num_results]
        except Exception:
            pass

    # 3. Both quotas exhausted → fallback to static
    return [
        "https://en.wikipedia.org/wiki/Cybersecurity",
        "https://www.cisa.gov/cybersecurity"
    ][:num_results]


def crawl_page(url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return ""
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])
        return text
    except:
        return ""

# ==============================
# ENDPOINT
# ==============================
@app.post("/answer")
def get_answer(data: QuestionRequest):
    try:
        question = data.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        search_urls = search_web(question)

        if not search_urls:
            return {"answer": "Sorry, I could not find any information."}

        # Crawl & summarize
        for url in search_urls:
            page_text = crawl_page(url)
            if page_text:
                try:
                    summary = get_summarizer()(page_text, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
                    return {"answer": summary}
                except Exception as e:
                    print(f"Summarization failed for {url}: {e}")
                    continue

        return {"answer": "Sorry, summarization failed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
