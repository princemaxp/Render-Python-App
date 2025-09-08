# render_app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import os

app = FastAPI(title="Guardian AI Render Service")

# HF Inference API token (set as secret on Render)
HF_TOKEN = os.getenv("HF_TOKEN")  # Use your HF_TOKEN secret

# Request schema
class QuestionRequest(BaseModel):
    question: str

@app.post("/answer")
def get_answer(data: QuestionRequest):
    try:
        question = data.question.strip()
        if not question:
            return {"answer": "Please provide a valid question."}

        # Search the web using SerpAPI (replace with your API key or other search API)
        search_urls = search_web(question)
        if not search_urls:
            return {"answer": "Sorry, no information was found."}

        # Crawl and summarize the first valid page
        summary = ""
        for url in search_urls:
            page_text = crawl_page(url)
            if page_text:
                summary = summarize_text(page_text)
                break

        if not summary:
            summary = "Sorry, no valid content could be extracted."

        return {"answer": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def search_web(query, num_results=3):
    """Search using SerpAPI or return fallback URLs."""
    SERP_API_KEY = os.getenv("SERP_API_KEY")  # optional, set on Render
    urls = []

    if SERP_API_KEY:
        try:
            params = {
                "q": query,
                "api_key": SERP_API_KEY,
                "num": num_results
            }
            response = requests.get("https://serpapi.com/search", params=params, timeout=5)
            results = response.json().get("organic_results", [])
            urls = [r["link"] for r in results if "link" in r]
        except:
            pass

    # Fallback URLs
    if not urls:
        urls = [
            "https://en.wikipedia.org/wiki/Cybersecurity",
            "https://www.cisa.gov/cybersecurity"
        ][:num_results]

    return urls


def crawl_page(url):
    """Get visible text from webpage."""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return ""
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = " ".join([p.get_text() for p in paragraphs])
        return text
    except:
        return ""


def summarize_text(text):
    """Summarize using HF Inference API to avoid loading large models."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": text, "parameters": {"max_new_tokens": 100}}

    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6",
            headers=headers,
            json=payload,
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and "summary_text" in result[0]:
                return result[0]["summary_text"]
            elif isinstance(result, dict) and "summary_text" in result:
                return result["summary_text"]
        return "Sorry, summarization failed."
    except:
        return "Sorry, summarization failed."
