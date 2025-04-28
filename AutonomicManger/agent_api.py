import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from .Agent import PegasusAgent

app = FastAPI(
    title="Pegasus Agent API",
    description="API for analyzing Pegasus workflow events with MAPE-K agent.",
    version="1.0.0"
)
agent = PegasusAgent()

class EventRequest(BaseModel):
    event: dict

@app.get("/", tags=["Health"])
def root():
    """Simple welcome endpoint."""
    return {"message": "Pegasus Agent API is running."}

@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint."""
    return {"status": "alive"}

@app.post("/analyze", tags=["Analysis"])
async def analyze_event(request: EventRequest):
    """
    Analyze a Pegasus workflow event.
    """
    result = await agent.analyze_event(request.event)
    return result

# Pour lancer : uvicorn agent_api:app --reload