# Generative Forecasting and Revenue Intelligence Platform

A production-grade AI platform combining **time series forecasting**, **graph neural networks**, **knowledge graphs**, and **retrieval-augmented generation (RAG)** for comprehensive revenue intelligence and forecasting.

## 🏗️ Architecture

```mermaid
graph TB
    A[Data Ingestion] --> B[Preprocessing Pipeline]
    B --> C[Time Series Models]
    B --> D[Graph Construction]
    B --> E[RAG Corpus]
    
    C --> F[LSTM Forecaster]
    C --> G[Transformer Forecaster]
    
    D --> H[PyTorch Geometric GNN]
    D --> I[Neo4j Knowledge Graph]
    
    E --> J[FAISS Dense Index]
    E --> K[BM25 Sparse Index]
    
    J --> L[Hybrid Retrieval]
    K --> L
    
    F --> M[FastAPI Server]
    G --> M
    H --> M
    I --> M
    L --> N[RAG Pipeline]
    N --> M
    
    M --> O[/forecast endpoint]
    M --> P[/rag-answer endpoint]
    M --> Q[/graph-insights endpoint]
```

## 🚀 Features

### 1. **Dual Forecasting Models**
- **LSTM**: Bidirectional LSTM with attention for sequential patterns
- **Transformer**: Multi-head attention for long-range dependencies
- Metrics: RMSE, MAE, MAPE
- Early stopping and model checkpointing

### 2. **Graph Neural Networks**
- Heterogeneous graph with stores, departments, and items
- GAT (Graph Attention Networks) and GraphSAGE support
- Revenue influence prediction
- Store similarity scoring

### 3. **Neo4j Knowledge Graph**
- Store and department metadata
- Relationship mapping (SELLS, SIMILAR_TO)
- Cypher query utilities
- Contextual insights for RAG

### 4. **RAG Pipeline**
- **Hybrid Retrieval**: Dense (FAISS) + Sparse (BM25)
- **LangChain Integration**: OpenAI GPT-4 for reasoning
- **Context Augmentation**: Neo4j + historical data
- Natural language Q&A

### 5. **FastAPI Server**
- RESTful API with automatic documentation
- Health checks and error handling
- CORS support
- Production-ready deployment

## 📁 Project Structure

```
walmart_dataset/
├── data/                      # Data files
│   ├── train.csv
│   ├── test.csv
│   ├── stores.csv
│   ├── features.csv
│   └── processed/            # Preprocessed data
├── ingestion/                # Data preprocessing
│   └── preprocess.py
├── forecasting/              # Time series models
│   ├── dataset.py
│   ├── model_lstm.py
│   ├── model_transformer.py
│   ├── train.py
│   ├── predict.py
│   └── models/              # Saved models
├── graph/                    # Graph neural networks
│   ├── build_graph.py
│   └── gnn_model.py
├── knowledge_graph/          # Neo4j integration
│   ├── neo4j_loader.py
│   └── neo4j_query.py
├── rag/                      # RAG pipeline
│   ├── build_corpus.py
│   ├── build_faiss.py
│   ├── hybrid_retrieval.py
│   ├── rag_pipeline.py
│   ├── corpus/              # Text corpus
│   └── faiss_index/         # Vector indices
├── api/                      # FastAPI server
│   ├── server.py
│   └── schemas.py
├── utils/                    # Utilities
│   ├── config.py
│   ├── logger.py
│   └── embedding.py
├── docker/                   # Docker configuration
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## 🛠️ Setup

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (for Neo4j)
- OpenAI API key (for RAG)
- 8GB+ RAM recommended

### Installation

1. **Clone and navigate to project**
```bash
cd walmart_dataset
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

5. **Start Neo4j (using Docker)**
```bash
cd docker
docker-compose up -d neo4j
```

## 📊 Data Preparation

### 1. Preprocess Data
```bash
python -m ingestion.preprocess
```

This will:
- Merge train/test/stores/features
- Create lag and rolling features
- Handle missing values
- Scale features
- Save to `data/processed/`

### 2. Build Graph
```bash
python -m graph.build_graph
```

Creates heterogeneous graph with:
- Store nodes with metadata
- Department nodes with sales stats
- Similarity edges based on features

### 3. Load Neo4j Knowledge Graph
```bash
python -m knowledge_graph.neo4j_loader
```

Populates Neo4j with:
- Store and department nodes
- SELLS relationships
- SIMILAR_TO relationships

### 4. Build RAG Corpus and Indices
```bash
# Build text corpus
python -m rag.build_corpus

# Build FAISS index
python -m rag.build_faiss

# Build hybrid retrieval (includes BM25)
python -m rag.hybrid_retrieval
```

## 🎯 Training Models

### Train LSTM Model
```bash
python -m forecasting.train
```

Or train specific model:
```python
from forecasting.train import train_lstm_model
from forecasting.dataset import split_train_val
import pandas as pd

data = pd.read_csv('data/processed/train_processed.csv')
train_data, val_data = split_train_val(data, val_split=0.2)
trainer = train_lstm_model(train_data, val_data)
```

### Train Transformer Model
```python
from forecasting.train import train_transformer_model
trainer = train_transformer_model(train_data, val_data)
```

Models are saved to `forecasting/models/`

## 🚀 Running the API

### Start FastAPI Server
```bash
python -m api.server
```

Or using uvicorn:
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

### Using Docker Compose (Full Stack)
```bash
cd docker
docker-compose up -d
```

This starts:
- Neo4j on ports 7474 (HTTP) and 7687 (Bolt)
- FastAPI on port 8000

### API Documentation
Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📡 API Endpoints

### 1. Forecast Revenue
```bash
POST /forecast
```

**Request:**
```json
{
  "store_id": 1,
  "dept_id": 1,
  "horizon": 4,
  "model_type": "lstm",
  "natural_language_query": "Explain the forecast drivers"
}
```

**Response:**
```json
{
  "store_id": 1,
  "dept_id": 1,
  "predictions": [15234.5, 15890.2, 16123.8, 15456.3],
  "forecast_dates": ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"],
  "model_type": "lstm",
  "explanation": "The forecast shows...",
  "metadata": {
    "last_date": "2023-12-25",
    "horizon": 4
  }
}
```

### 2. RAG Question Answering
```bash
POST /rag-answer
```

**Request:**
```json
{
  "question": "What are the top performing stores and why?",
  "store_id": 1,
  "top_k": 5
}
```

**Response:**
```json
{
  "question": "What are the top performing stores and why?",
  "answer": "Based on the data, the top performing stores are...",
  "sources": [
    {
      "text": "Store 1 Summary: Type A, Size 150,000...",
      "metadata": {"store_id": 1, "type": "store_summary"},
      "score": 0.89
    }
  ]
}
```

### 3. Graph Insights
```bash
POST /graph-insights
```

**Request:**
```json
{
  "store_id": 1,
  "include_similar_stores": true,
  "include_departments": true
}
```

**Response:**
```json
{
  "store_id": 1,
  "store_info": {
    "store_id": 1,
    "type": "A",
    "size": 151315
  },
  "departments": [
    {
      "dept_id": 1,
      "avg_sales": 24924.5,
      "total_sales": 1871587.67
    }
  ],
  "similar_stores": [
    {
      "store_id": 13,
      "type": "A",
      "size": 155078,
      "similarity": 0.8
    }
  ]
}
```

### 4. Health Check
```bash
GET /health
```

## 🧪 Example Usage

### Python Client
```python
import requests

# Forecast
response = requests.post('http://localhost:8000/forecast', json={
    'store_id': 1,
    'dept_id': 1,
    'horizon': 4,
    'model_type': 'transformer',
    'natural_language_query': 'Explain the forecast'
})
print(response.json())

# RAG Q&A
response = requests.post('http://localhost:8000/rag-answer', json={
    'question': 'Which departments have the highest sales?',
    'top_k': 3
})
print(response.json()['answer'])

# Graph Insights
response = requests.post('http://localhost:8000/graph-insights', json={
    'store_id': 1,
    'include_similar_stores': True
})
print(response.json())
```

### cURL
```bash
# Forecast
curl -X POST "http://localhost:8000/forecast" \
  -H "Content-Type: application/json" \
  -d '{"store_id": 1, "dept_id": 1, "horizon": 4, "model_type": "lstm"}'

# RAG Answer
curl -X POST "http://localhost:8000/rag-answer" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the sales trends?"}'
```

## 🔧 Configuration

Edit `utils/config.py` to customize:
- Model architectures (LSTM, Transformer, GNN)
- Training hyperparameters
- RAG retrieval weights
- API settings

## 📈 Model Performance

Expected metrics (will vary based on data):
- **LSTM**: RMSE ~2000-3000, MAPE ~15-20%
- **Transformer**: RMSE ~1800-2800, MAPE ~14-18%

## 🐛 Troubleshooting

### Neo4j Connection Issues
```bash
# Check Neo4j is running
docker ps | grep neo4j

# View logs
docker logs revenue_neo4j
```

### FAISS Index Not Found
```bash
# Rebuild indices
python -m rag.build_corpus
python -m rag.build_faiss
```

### OpenAI API Errors
- Verify API key in `.env`
- Check quota and billing

## 🚢 Production Deployment

### Using Docker Compose
```bash
cd docker
docker-compose up -d
```

### Environment Variables
Set in `.env`:
- `OPENAI_API_KEY`
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- `API_HOST`, `API_PORT`

### Scaling
- Use Redis for caching (uncomment in docker-compose.yml)
- Deploy behind nginx/traefik for load balancing
- Use GPU for faster inference

## 📚 Technologies

- **PyTorch**: Deep learning models
- **PyTorch Geometric**: Graph neural networks
- **LangChain**: RAG orchestration
- **FAISS**: Vector similarity search
- **Neo4j**: Knowledge graph database
- **FastAPI**: High-performance API
- **OpenAI GPT-4**: Language model reasoning

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

## 📧 Contact

For questions or support, please open an issue.

---

**Built with ❤️ for production-grade revenue intelligence**
