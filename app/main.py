from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.api import routes
from app.utils.logger import setup_logging
import logging

# Load environmenta variables
load_dotenv()

# Setup logging
setup_logging()
logging.getLogger("watchfiles").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS configuration to allow cross-origin requests from Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your Streamlit frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router)

@app.on_event("startup")
async def startup_event():
    logger.info("RAG Application starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("RAG Application shutting down...")

@app.get("/")
def read_root():
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to the FastAPI backend for the Streamlit app!"}