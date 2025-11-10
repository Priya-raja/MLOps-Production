import os
from typing import Generator, Iterable
from fastapi import FastAPI, Depends, HTTPException, status  # type: ignore
from fastapi.responses import StreamingResponse  # type: ignore
from fastapi_clerk_auth import ClerkConfig, ClerkHTTPBearer, HTTPAuthorizationCredentials  # type: ignore
from openai import OpenAI  # type: ignore

app = FastAPI()

clerk_config = ClerkConfig(jwks_url=os.getenv("CLERK_JWKS_URL"))
clerk_guard = ClerkHTTPBearer(clerk_config)

@app.get("/api")
def idea(creds: HTTPAuthorizationCredentials = Depends(clerk_guard)):
    user_id = creds.decoded["sub"]  # User ID from JWT - available for future use
    # You could use user_id to track usage, store ideas, apply limits, etc.

    # Ensure OpenAI is configured
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OpenAI API key not configured",
        )

    # Allow overriding the model via env; default to a broadly available model
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    try:
        client = OpenAI()
        prompt = [
            {
                "role": "user",
                "content": "Reply with a new business idea for AI Agents, formatted with headings, sub-headings and bullet points",
            }
        ]
        stream: Iterable = client.chat.completions.create(
            model=model,
            messages=prompt,
            stream=True,
        )
    except Exception as e:
        # Surface useful information without leaking sensitive details
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start OpenAI stream: {str(e)}",
        )

    def event_stream():
        for chunk in stream:
            text = chunk.choices[0].delta.content
            if text:
                lines = text.split("\n")
                for line in lines[:-1]:
                    yield f"data: {line}\n\n"
                    yield "data:  \n"
                yield f"data: {lines[-1]}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")