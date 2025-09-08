# render_app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import os
import re
from collections import Counter

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

class QuestionRequest(BaseModel):
    question: str

# ==============================
# HEALTH
# ==============================
@app.get("/")
def root():
    return {"status": "ok", "service": "Guardian AI Render Service"}

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
        response = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            return ""
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove scripts/styles/navs
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()

        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
        # Normalize whitespace & trim to keep CPU low
        text = re.sub(r"\s+", " ", text).strip()
        return text[:12000]  # safety cap
    except Exception:
        return ""

# ==============================
# SUMMARIZATION (no NLTK / no HF)
# ==============================
STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being below
between both but by could did do does doing down during each few for from further had has have
having he her here hers herself him himself his how i if in into is it its itself just me more
most my myself no nor not of off on once only or other ought our ours ourselves out over own same
she should so some such than that the their theirs them themselves then there these they this those
through to too under until up very was we were what when where which while who whom why with would
you your yours yourself yourselves
""".split())

def split_sentences(text: str):
    # Simple sentence splitter without NLTK
    # Keep punctuation; split on ., !, ?
    parts = re.split(r'(?<=[\.\!\?])\s+', text)
    # Clean & discard tiny fragments
    return [s.strip() for s in parts if len(s.strip().split()) >= 5]

def tokenize_words(text: str):
    return re.findall(r"[a-zA-Z']{2,}", text.lower())

def clean_summary(summary: str) -> str:
    """Remove irrelevant boilerplate like author/job titles from the summary."""
    sentences = split_sentences(summary)
    filtered = []
    for s in sentences:
        # Skip common boilerplate/author lines
        if re.search(r"(Staff Writer|Editorial|Content Lead|Reporter|Journalist|Subscribe|Sign up)", s, re.IGNORECASE):
            continue
        filtered.append(s)
    return " ".join(filtered)

def summarize_text(text: str, max_sentences: int = 5) -> str:
    try:
        sentences = split_sentences(text)
        if not sentences:
            return ""

        # For speed, consider only first N sentences when pages are huge
        sentences = sentences[:80]

        # Build word frequency (ignore stopwords)
        words = tokenize_words(" ".join(sentences))
        words = [w for w in words if w not in STOPWORDS]
        if not words:
            return " ".join(sentences[:max_sentences])

        freqs = Counter(words)
        max_freq = max(freqs.values())
        # Normalize to 0..1
        for w in list(freqs.keys()):
            freqs[w] = freqs[w] / max_freq

        # Score sentences
        sent_scores = []
        for idx, s in enumerate(sentences):
            tokens = tokenize_words(s)
            if not tokens:
                continue
            score = sum(freqs.get(t, 0.0) for t in tokens) / (len(tokens) ** 0.8)  # slight length penalty
            sent_scores.append((idx, score, s))

        # Pick top K by score, then restore original order
        top = sorted(sent_scores, key=lambda x: x[1], reverse=True)[:max_sentences]
        top_sorted = [s for _, _, s in sorted(top, key=lambda x: x[0])]

        # Clean boilerplate
        summary = clean_summary(" ".join(top_sorted)).strip()

        # If cleaning removed everything, fallback to first K sentences
        if not summary:
            summary = " ".join(sentences[:max_sentences])

        # Final tidy
        summary = re.sub(r"\s+", " ", summary).strip()
        return summary
    except Exception as e:
        print(f"Summarization failed: {e}")
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
                summary = summarize_text(page_text, max_sentences=5)
                if summary:
                    return {"answer": summary}

        return {"answer": "Sorry, summarization failed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
