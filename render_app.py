# render_app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

app = FastAPI(title="Guardian AI Render Service")

# Define request schema
class QuestionRequest(BaseModel):
    question: str

# Initialize summarizer
summarizer = pipeline("summarization")

@app.post("/answer")
def get_answer(data: QuestionRequest):
    try:
        question = data.question
        # Example: simple search on Google via API or custom logic
        search_urls = search_web(question)
        if not search_urls:
            return {"answer": "Sorry, I could not find any information."}

        # Crawl and summarize first valid page
        summary = ""
        for url in search_urls:
            page_text = crawl_page(url)
            if page_text:
                summary = summarizer(page_text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
                break

        if not summary:
            summary = "Sorry, no valid content could be extracted."

        return {"answer": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def search_web(query, num_results=3):
    # You can replace this with a proper Google Search API or Bing API
    # For now, return some hardcoded URLs for testing
    return [
        "https://en.wikipedia.org/wiki/Cybersecurity",
        "https://www.cisa.gov/cybersecurity"
    ][:num_results]

def crawl_page(url):
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
