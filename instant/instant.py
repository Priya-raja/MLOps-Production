from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from openai import OpenAI

import os

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def instant():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return "<html><body><p>Error: OPENAI_API_KEY not set</p></body></html>"
    
    client = OpenAI(api_key=api_key)
    message = """
You are on a website that has just been deployed to production for the first time!
Please reply with an enthusiastic announcement to welcome visitors to the site, explaining that it is live on production for the first time!
"""
    messages = [{"role": "user", "content": message}]
    response = client.chat.completions.create(model="gpt-5-nano", messages=messages)
    reply = response.choices[0].message.content.replace("\n", "<br/>")
    html = f"<html><head><title>Live in an Instant!</title></head><body><p>{reply}</p></body></html>"
    return html