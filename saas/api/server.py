import os
import logging
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi_clerk_auth import ClerkConfig, ClerkHTTPBearer, HTTPAuthorizationCredentials
from openai import OpenAI
from starlette.middleware.base import BaseHTTPMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Middleware to log authentication attempts
class AuthLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/api/"):
            auth_header = request.headers.get("Authorization", "Not provided")
            auth_present = "Bearer" in auth_header if auth_header != "Not provided" else False
            logger.info(f"API request to {request.url.path} - Auth header present: {auth_present}")
        response = await call_next(request)
        return response

app.add_middleware(AuthLoggingMiddleware)

# Add CORS middleware (allows frontend to call backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clerk authentication setup
clerk_jwks_url = os.getenv("CLERK_JWKS_URL")
if not clerk_jwks_url:
    logger.error("CLERK_JWKS_URL environment variable is not set! Authentication will fail.")
    raise ValueError("CLERK_JWKS_URL environment variable must be set")

clerk_config = ClerkConfig(jwks_url=clerk_jwks_url)
clerk_guard = ClerkHTTPBearer(clerk_config)

# Exception handler for authentication errors
@app.exception_handler(HTTPException)
async def auth_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == 403 or exc.status_code == 401:
        logger.error(f"Authentication failed for {request.url.path}: {exc.detail}")
        logger.error(f"CLERK_JWKS_URL: {clerk_jwks_url[:50]}..." if clerk_jwks_url else "CLERK_JWKS_URL: Not set")
        auth_header = request.headers.get("Authorization", "Not provided")
        logger.error(f"Authorization header present: {bool(auth_header and auth_header != 'Not provided')}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.on_event("startup")
async def startup_event():
    """Log configuration on startup"""
    logger.info("Starting FastAPI server...")
    logger.info(f"CLERK_JWKS_URL is set: {bool(clerk_jwks_url)}")
    if clerk_jwks_url:
        logger.info(f"CLERK_JWKS_URL starts with: {clerk_jwks_url[:50]}...")
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY environment variable is not set!")

class Visit(BaseModel):
    patient_name: str
    date_of_visit: str
    notes: str

system_prompt = """
You are provided with notes written by a doctor from a patient's visit.
Your job is to summarize the visit for the doctor and provide an email.
Reply with exactly three sections with the headings:
### Summary of visit for the doctor's records
### Next steps for the doctor
### Draft of email to patient in patient-friendly language
"""

def user_prompt_for(visit: Visit) -> str:
    return f"""Create the summary, next steps and draft email for:
Patient Name: {visit.patient_name}
Date of Visit: {visit.date_of_visit}
Notes:
{visit.notes}"""

@app.post("/api/consultation")
def consultation_summary(
    visit: Visit,
    creds: HTTPAuthorizationCredentials = Depends(clerk_guard),
):
    try:
        user_id = creds.decoded["sub"]
        logger.info(f"Processing consultation request for user: {user_id}")
    except Exception as e:
        logger.error(f"Failed to decode JWT token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token"
        )
    
    try:
        client = OpenAI()
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OpenAI API key not configured"
        )
    
    user_prompt = user_prompt_for(visit)
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",  # Fixed: gpt-5-nano doesn't exist, use gpt-4o-mini
            messages=prompt,
            stream=True,
        )
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate consultation summary: {str(e)}"
        )
    
    def event_stream():
        try:
            for chunk in stream:
                text = chunk.choices[0].delta.content
                if text:
                    lines = text.split("\n")
                    for line in lines[:-1]:
                        yield f"data: {line}\n\n"
                        yield "data:  \n"
                    yield f"data: {lines[-1]}\n\n"
        except Exception as e:
            logger.error(f"Error in event stream: {e}")
            yield f"data: Error: {str(e)}\n\n"
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/health")
def health_check():
    """Health check endpoint for AWS App Runner"""
    return {"status": "healthy"}

# Serve static files (our Next.js export) - MUST BE LAST!
static_path = Path("static")
if static_path.exists():
    # Serve index.html for the root path
    @app.get("/")
    async def serve_root():
        return FileResponse(static_path / "index.html")
    
    # Mount static files for all other routes
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
