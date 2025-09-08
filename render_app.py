# render_app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# ✅ New: lightweight summarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

app = FastAPI(title="Guardian AI Render Service")

# ==============================
# CONFIG
# ==============================
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
GOOGLE_CX = "YOUR_GOOGLE_CX_ID"
SERP_API_KEY = "YOUR_SERP_API_KEY"

GOOGLE_LIMIT = 100
SERP_LIMIT = 3

# Usage trackers
usage = {"google": 0, "serp": 0}
reset_time = datetime.now() + timedelta(days=1)

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
    reset_usage_if_needed()
    urls = []

    # 1. Google API
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

    # 3. Static fallback
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

# ✅ Replace HF with sumy
def summarize_text(text, sentence_count=3):
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, sentence_count)
        return " ".join([str(sentence) for sentence in summary])
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
                summary = summarize_text(page_text)
                if summary:
                    return {"answer": summary}

        return {"answer": "Sorry, summarization failed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
