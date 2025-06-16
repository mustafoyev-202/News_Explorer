"""
News Explorer - A clean, scalable news search application
Author: AI Assistant
Version: 1.0.0
"""

import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from typing import List, Optional
from datetime import datetime

import certifi
import pandas as pd
import pymongo
import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
class Config:
    """Application configuration"""
    MONGO_URI: str = os.getenv("MONGO_URI", "")
    KAGGLE_API_KEY: str = os.getenv("KAGGLE_API_KEY", "")
    DATABASE_NAME: str = "news_db"
    COLLECTION_NAME: str = "articles"
    MAX_ARTICLES: int = 1000
    SEARCH_LIMIT: int = 10
    KAGGLE_DATASET_URL: str = "https://www.kaggle.com/api/v1/datasets/download/uciml/news-aggregator-dataset"
    
    @classmethod
    def validate(cls) -> None:
        """Validate required environment variables"""
        if not cls.MONGO_URI:
            raise ValueError("MONGO_URI environment variable is required")
        if not cls.KAGGLE_API_KEY:
            raise ValueError("KAGGLE_API_KEY environment variable is required")

# Pydantic models
class SearchQuery(BaseModel):
    """Search query model"""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")

class Article(BaseModel):
    """Article model"""
    title: str
    publication: str
    date: datetime
    category: str
    url: Optional[str] = None

class SearchResult(BaseModel):
    """Search result model"""
    title: str
    publication: str
    date: datetime
    category: str
    url: Optional[str] = None
    score: float

class SearchResponse(BaseModel):
    """Search response model"""
    results: List[SearchResult]
    summary: str
    total_results: int

# Database service
class DatabaseService:
    """Database service for MongoDB operations"""
    
    def __init__(self, uri: str, db_name: str, collection_name: str):
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
    
    async def connect(self) -> None:
        """Connect to MongoDB"""
        try:
            self.client = pymongo.MongoClient(
                self.uri,
                serverSelectionTimeoutMS=30000,
                connectTimeoutMS=30000,
                socketTimeoutMS=30000,
                maxPoolSize=10,
                retryWrites=True,
                retryReads=True,
                tls=True,
                tlsAllowInvalidCertificates=False,
                tlsAllowInvalidHostnames=False,
                tlsCAFile=certifi.where(),
            )
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            logger.info("MongoDB connection successful")
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database connection failed"
            )
    
    async def disconnect(self) -> None:
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    async def clear_collection(self) -> None:
        """Clear all documents from collection"""
        try:
            self.collection.delete_many({})
            logger.info("Collection cleared")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to clear collection"
            )
    
    async def insert_articles(self, articles: List[dict]) -> None:
        """Insert articles into collection"""
        try:
            self.collection.insert_many(articles)
            logger.info(f"Inserted {len(articles)} articles")
        except Exception as e:
            logger.error(f"Error inserting articles: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to insert articles"
            )
    
    async def create_text_index(self) -> None:
        """Create text index on title field"""
        try:
            # Drop existing index if exists
            try:
                self.collection.drop_index("title_text_index")
                logger.info("Dropped existing text index")
            except Exception as e:
                logger.info(f"No existing index to drop: {e}")
            
            # Create new text index with longer timeout
            logger.info("Creating text index...")
            self.collection.create_index(
                [("title", pymongo.TEXT)],
                name="title_text_index",
                background=True  # Create index in background
            )
            logger.info("Text index created successfully")
        except Exception as e:
            logger.error(f"Error creating text index: {e}")
            # Don't fail the application if index creation fails
            logger.warning("Continuing without text index - search functionality may be limited")
    
    async def search_articles(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search articles by text query"""
        try:
            # Check if text index exists
            indexes = self.collection.list_indexes()
            has_text_index = any(index.get('name') == 'title_text_index' for index in indexes)
            
            if has_text_index:
                # Use text search if index exists
                pipeline = [
                    {
                        "$match": {
                            "$text": {"$search": query}
                        }
                    },
                    {
                        "$sort": {
                            "score": {"$meta": "textScore"}
                        }
                    },
                    {
                        "$limit": limit
                    },
                    {
                        "$project": {
                            "title": 1,
                            "publication": 1,
                            "date": 1,
                            "category": 1,
                            "url": 1,
                            "score": {"$meta": "textScore"},
                            "_id": 0
                        }
                    }
                ]
            else:
                # Fallback to regex search if no text index
                pipeline = [
                    {
                        "$match": {
                            "title": {"$regex": query, "$options": "i"}
                        }
                    },
                    {
                        "$limit": limit
                    },
                    {
                        "$project": {
                            "title": 1,
                            "publication": 1,
                            "date": 1,
                            "category": 1,
                            "url": 1,
                            "score": {"$literal": 1.0},
                            "_id": 0
                        }
                    }
                ]
            
            results = list(self.collection.aggregate(pipeline))
            return [SearchResult(**result) for result in results]
        except Exception as e:
            logger.error(f"Error searching articles: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Search failed"
            )

# Data service
class DataService:
    """Service for handling data operations"""
    
    def __init__(self, kaggle_api_key: str, dataset_url: str):
        self.kaggle_api_key = kaggle_api_key
        self.dataset_url = dataset_url
    
    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy"""
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        session.mount("https://", HTTPAdapter(max_retries=retries))
        return session
    
    async def download_dataset(self) -> pd.DataFrame:
        """Download and extract dataset from Kaggle"""
        try:
            headers = {"Authorization": f"Bearer {self.kaggle_api_key}"}
            session = self._create_session()
            
            logger.info("Downloading dataset...")
            response = session.get(self.dataset_url, headers=headers, timeout=30)
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to download dataset: {response.status_code}"
                )
            
            # Extract CSV from ZIP
            import zipfile
            import io
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                csv_files = [f for f in zip_ref.namelist() if f.endswith(".csv")]
                if not csv_files:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="No CSV files found in archive"
                    )
                
                csv_filename = csv_files[0]
                logger.info(f"Found CSV file: {csv_filename}")
                
                with zip_ref.open(csv_filename) as csv_file:
                    df = pd.read_csv(csv_file)
                    logger.info(f"Loaded {len(df)} rows from CSV")
                    return df
                    
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Dataset download failed: {str(e)}"
            )
    
    def process_dataframe(self, df: pd.DataFrame) -> List[dict]:
        """Process and clean dataframe"""
        try:
            # Validate required columns
            required_columns = ["TITLE", "PUBLISHER", "TIMESTAMP", "CATEGORY", "URL"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing columns: {missing_columns}")
            
            # Convert timestamps
            logger.info("Converting timestamps...")
            df["date"] = pd.to_datetime(df["TIMESTAMP"] // 1000, unit="s")
            
            # Map categories
            category_map = {
                "b": "business",
                "t": "technology", 
                "e": "entertainment",
                "m": "health"
            }
            
            df["category"] = df["CATEGORY"].map(category_map)
            df = df.dropna(subset=["category"])
            
            # Prepare articles
            articles = (
                df[["TITLE", "PUBLISHER", "date", "category", "URL"]]
                .rename(columns={
                    "TITLE": "title",
                    "PUBLISHER": "publication",
                    "URL": "url"
                })
                .to_dict("records")
            )
            
            logger.info(f"Processed {len(articles)} articles")
            return articles
            
        except Exception as e:
            logger.error(f"Error processing dataframe: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Data processing failed: {str(e)}"
            )

# Application service
class NewsExplorerService:
    """Main application service"""
    
    def __init__(self):
        self.db_service = DatabaseService(
            Config.MONGO_URI,
            Config.DATABASE_NAME,
            Config.COLLECTION_NAME
        )
        self.data_service = DataService(
            Config.KAGGLE_API_KEY,
            Config.KAGGLE_DATASET_URL
        )
    
    async def initialize(self) -> None:
        """Initialize the application"""
        await self.db_service.connect()
        await self.load_dataset()
    
    async def load_dataset(self) -> None:
        """Load dataset into database"""
        try:
            # Download and process dataset
            df = await self.data_service.download_dataset()
            articles = self.data_service.process_dataframe(df)
            
            # Limit articles for performance
            articles = articles[:Config.MAX_ARTICLES]
            logger.info(f"Limited to {len(articles)} articles")
            
            # Clear and populate database
            await self.db_service.clear_collection()
            await self.db_service.insert_articles(articles)
            
            # Try to create text index, but don't fail if it times out
            try:
                await self.db_service.create_text_index()
            except Exception as e:
                logger.warning(f"Text index creation failed: {e}")
                logger.info("Application will continue with basic search functionality")
            
            logger.info("Dataset loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    async def search(self, query: str) -> SearchResponse:
        """Search articles"""
        try:
            results = await self.db_service.search_articles(
                query, 
                Config.SEARCH_LIMIT
            )
            
            # Create summary
            summary_text = " ".join([r.title for r in results])
            summary = summary_text[:200] + "..." if len(summary_text) > 200 else summary_text
            summary = summary if results else "No results found."
            
            return SearchResponse(
                results=results,
                summary=summary,
                total_results=len(results)
            )
            
        except Exception as e:
            logger.error(f"Error in search: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        await self.db_service.disconnect()

# Global service instance
news_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global news_service
    
    # Startup
    try:
        # Validate configuration
        Config.validate()
        
        # Initialize service
        news_service = NewsExplorerService()
        await news_service.initialize()
        
        logger.info("Application started successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        sys.exit(1)
    
    yield
    
    # Shutdown
    if news_service:
        await news_service.cleanup()
        logger.info("Application shutdown complete")

# FastAPI application
app = FastAPI(
    title="News Explorer",
    description="A clean, scalable news search application",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Signal handlers
def signal_handler(sig, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal, exiting gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# API endpoints
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search", response_model=SearchResponse)
async def search(query: SearchQuery):
    """Search articles endpoint"""
    try:
        return await news_service.search(query.query)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

if __name__ == "__main__":
    try:
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8500,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        sys.exit(1) 