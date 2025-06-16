# News Explorer

A clean, scalable, and maintainable news search application built with FastAPI and MongoDB.

## Features

- 🔍 **Advanced Search**: Full-text search across news article titles
- 📊 **Real-time Results**: Instant search results with relevance scoring
- 🎨 **Modern UI**: Clean, responsive design with smooth animations
- 🚀 **High Performance**: Optimized database queries and indexing
- 🔒 **Secure**: Proper error handling and input validation
- 📱 **Mobile Friendly**: Responsive design for all devices

## Architecture

The application follows clean architecture principles with proper separation of concerns:

- **Service Layer**: Business logic separated into dedicated services
- **Data Layer**: MongoDB integration with proper connection management
- **API Layer**: FastAPI endpoints with Pydantic validation
- **UI Layer**: Modern, responsive frontend with vanilla JavaScript

## Tech Stack

- **Backend**: FastAPI, Python 3.8+
- **Database**: MongoDB
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Styling**: Custom CSS with modern design patterns
- **Icons**: Font Awesome
- **Data Source**: Kaggle News Aggregator Dataset

## Prerequisites

- Python 3.8 or higher
- MongoDB Atlas account (or local MongoDB)
- Kaggle API key

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd news-explorer
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   MONGO_URI=your_mongodb_connection_string
   KAGGLE_API_KEY=your_kaggle_api_key
   ```

5. **Run the application**
   ```bash
   python news_explorer.py
   ```

6. **Access the application**
   Open your browser and navigate to `http://127.0.0.1:8500`

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `MONGO_URI` | MongoDB connection string | Yes |
| `KAGGLE_API_KEY` | Kaggle API key for dataset access | Yes |

### Application Settings

The application includes configurable settings in the `Config` class:

- `MAX_ARTICLES`: Maximum number of articles to load (default: 1000)
- `SEARCH_LIMIT`: Maximum search results (default: 10)
- `DATABASE_NAME`: MongoDB database name (default: "news_db")
- `COLLECTION_NAME`: MongoDB collection name (default: "articles")

## API Endpoints

### GET `/`
- **Description**: Home page with search interface
- **Response**: HTML page

### POST `/search`
- **Description**: Search articles by query
- **Request Body**:
  ```json
  {
    "query": "search term"
  }
  ```
- **Response**:
  ```json
  {
    "results": [
      {
        "title": "Article Title",
        "publication": "Publication Name",
        "date": "2023-01-01T00:00:00",
        "category": "technology",
        "url": "https://example.com",
        "score": 1.5
      }
    ],
    "summary": "Search summary...",
    "total_results": 10
  }
  ```

### GET `/health`
- **Description**: Health check endpoint
- **Response**:
  ```json
  {
    "status": "healthy",
    "timestamp": "2023-01-01T00:00:00"
  }
  ```

## Database Schema

### Articles Collection

```json
{
  "_id": "ObjectId",
  "title": "string",
  "publication": "string",
  "date": "datetime",
  "category": "string",
  "url": "string"
}
```

### Indexes

- **Text Index**: `title_text_index` on the `title` field for full-text search

## Error Handling

The application includes comprehensive error handling:

- **Input Validation**: Pydantic models validate all inputs
- **Database Errors**: Proper MongoDB error handling and connection management
- **Network Errors**: Retry logic for external API calls
- **User Feedback**: Clear error messages in the UI

## Performance Optimizations

- **Database Indexing**: Text index on article titles for fast search
- **Connection Pooling**: Optimized MongoDB connection settings
- **Query Optimization**: Efficient aggregation pipelines
- **Frontend Caching**: Minimal API calls with proper error handling

## Security Features

- **Input Sanitization**: All user inputs are validated and sanitized
- **Error Information**: Limited error details exposed to users
- **Connection Security**: TLS/SSL for database connections
- **Rate Limiting**: Built-in request handling

## Development

### Code Structure

```
news-explorer/
├── news_explorer.py      # Main application file
├── templates/
│   └── index.html        # Frontend template
├── static/               # Static assets
├── requirements.txt      # Python dependencies
├── README.md            # Documentation
└── .env                 # Environment variables
```

### Key Components

1. **Config Class**: Centralized configuration management
2. **DatabaseService**: MongoDB operations and connection management
3. **DataService**: Dataset processing and external API calls
4. **NewsExplorerService**: Main application logic
5. **FastAPI App**: API endpoints and request handling

### Best Practices Implemented

- **Separation of Concerns**: Each class has a single responsibility
- **Error Handling**: Comprehensive exception handling throughout
- **Logging**: Structured logging for debugging and monitoring
- **Type Hints**: Full type annotations for better code quality
- **Documentation**: Clear docstrings and comments
- **Configuration**: Environment-based configuration
- **Testing Ready**: Clean architecture for easy testing

## Deployment

### Docker Deployment

1. **Build the image**
   ```bash
   docker build -t news-explorer .
   ```

2. **Run the container**
   ```bash
   docker run -p 8500:8500 --env-file .env news-explorer
   ```

### Production Considerations

- Use a production WSGI server (Gunicorn)
- Set up proper logging and monitoring
- Configure environment variables securely
- Set up database backups and monitoring
- Use a reverse proxy (Nginx) for production

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For support and questions, please open an issue in the repository. 