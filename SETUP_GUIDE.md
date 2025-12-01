# Revenue Intelligence Platform - Step-by-Step Setup Guide

## 📋 Prerequisites

Before you begin, ensure you have:
- ✅ Python 3.13 installed
- ✅ Docker Desktop installed and running
- ✅ Git (optional, for version control)
- ✅ NVIDIA GPU with CUDA support (optional, for GPU acceleration)
- ✅ OpenAI API key (for RAG functionality)

## 🚀 Quick Start (5 Steps)

### Step 1: Install Dependencies
```bash
cd e:\SquarkAI\walmart_dataset
pip install -r requirements.txt
```

### Step 2: Install PyTorch with CUDA
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
```

### Step 3: Start Neo4j Database
```bash
cd docker
docker-compose up -d neo4j
```

### Step 4: Configure Environment
Create `.env` file:
```bash
OPENAI_API_KEY=your_api_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

### Step 5: Run the Platform
```bash
# Terminal 1: Start API Server
python -m api.server

# Terminal 2: Start Web Interface
python -m http.server 3000 --directory web
```

**Done!** Access at http://localhost:3000

---

## 📖 Detailed Setup Guide

### Part 1: Environment Setup

#### 1.1 Clone/Navigate to Project
```bash
cd e:\SquarkAI\walmart_dataset
```

#### 1.2 Create Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

#### 1.3 Install Python Dependencies
```bash
pip install -r requirements.txt
```

**Expected output**: ~30 packages installed including:
- fastapi
- pandas
- numpy
- scikit-learn
- neo4j
- openai
- langchain
- etc.

#### 1.4 Install PyTorch with CUDA Support
```bash
# For NVIDIA RTX 4050 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
```

**Verify installation**:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
# Expected: CUDA: True
```

### Part 2: Database Setup

#### 2.1 Start Docker Desktop
- Open Docker Desktop application
- Wait for it to be running (green icon)

#### 2.2 Start Neo4j Container
```bash
cd docker
docker-compose up -d neo4j
```

**Expected output**:
```
Creating revenue_neo4j ... done
```

#### 2.3 Verify Neo4j is Running
```bash
docker ps
```

**Expected**: You should see `revenue_neo4j` container running

**Access Neo4j Browser**: http://localhost:7474
- Username: `neo4j`
- Password: `password`

### Part 3: Configuration

#### 3.1 Create Environment File
Create `.env` file in project root:
```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

**Get OpenAI API Key**: https://platform.openai.com/api-keys

#### 3.2 Verify Configuration
```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('API Key loaded:', bool(os.getenv('OPENAI_API_KEY')))"
# Expected: API Key loaded: True
```

### Part 4: Data Pipeline (First Time Only)

#### 4.1 Run Data Ingestion
```bash
python main.py --steps ingest
```

**What this does**:
1. Loads raw Walmart data
2. Preprocesses and engineers features
3. Builds knowledge graph
4. Loads data into Neo4j
5. Creates RAG indices (FAISS + BM25)

**Expected time**: 2-3 minutes

**Expected output**:
```
✓ Data preprocessing complete
✓ Graph built
✓ Neo4j loaded
✓ RAG indices created
```

#### 4.2 Train Models (First Time Only)
```bash
python main.py --steps train
```

**What this does**:
1. Trains LSTM forecasting model
2. Trains Transformer forecasting model
3. Saves models to `forecasting/models/`

**Expected time**: 
- GPU: ~18 minutes
- CPU: ~80 minutes

**Expected output**:
```
Training LSTM model...
✓ LSTM model trained (23 epochs)
Training Transformer model...
✓ Transformer model trained (11 epochs)
```

### Part 5: Running the Platform

#### 5.1 Start API Server
Open **Terminal 1**:
```bash
cd e:\SquarkAI\walmart_dataset
python -m api.server
```

**Expected output**:
```
INFO: Uvicorn running on http://0.0.0.0:8000
INFO: Application startup complete.
```

**Keep this terminal running!**

#### 5.2 Start Web Interface
Open **Terminal 2**:
```bash
cd e:\SquarkAI\walmart_dataset
python -m http.server 3000 --directory web
```

**Expected output**:
```
Serving HTTP on :: port 3000 (http://[::]:3000/) ...
```

**Keep this terminal running!**

### Part 6: Access the Platform

#### 6.1 Web Interface (Main Access Point)
**URL**: http://localhost:3000

**Features**:
- AI chatbox for natural language queries
- Example command cards
- Real-time forecasting
- Store similarity analysis

#### 6.2 API Documentation
**URL**: http://localhost:8000/docs

**Features**:
- Interactive API documentation
- Try endpoints directly
- View request/response schemas

#### 6.3 Neo4j Browser
**URL**: http://localhost:7474

**Login**:
- Username: `neo4j`
- Password: `password`

**Features**:
- Visual graph exploration
- Run Cypher queries
- View store relationships

## 🎯 Usage Examples

### Example 1: Get Revenue Forecast
**Via Web Interface**:
1. Go to http://localhost:3000
2. Type: "Forecast revenue for store 1 department 1"
3. Click send or press Enter

**Via API**:
```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "store_id": 1,
    "dept_id": 1,
    "horizon": 4,
    "model_type": "lstm"
  }'
```

### Example 2: Ask Natural Language Questions
**Via Web Interface**:
1. Go to http://localhost:3000
2. Type: "Which stores are similar to store 5?"
3. Get AI-powered answer with sources

**Via API**:
```bash
curl -X POST http://localhost:8000/rag-answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Which stores are similar to store 5?"
  }'
```

### Example 3: Explore Graph Data
**Via Neo4j Browser**:
1. Go to http://localhost:7474
2. Login with neo4j/password
3. Run query:
```cypher
MATCH (s:Store {store_id: 1})-[:SIMILAR_TO]->(similar)
RETURN s, similar
```

## 🔧 Troubleshooting

### Issue 1: "Port already in use"
**Problem**: Port 8000 or 3000 already in use

**Solution**:
```bash
# Find process using port
netstat -ano | findstr :8000

# Kill process (replace PID)
taskkill /PID <PID> /F
```

### Issue 2: "CUDA not available"
**Problem**: PyTorch not detecting GPU

**Solution**:
```bash
# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue 3: "Neo4j connection failed"
**Problem**: Can't connect to Neo4j

**Solution**:
```bash
# Restart Neo4j
docker-compose restart neo4j

# Check logs
docker logs revenue_neo4j
```

### Issue 4: "OpenAI API quota exceeded"
**Problem**: 429 error from OpenAI

**Solution**:
1. Go to https://platform.openai.com/account/billing
2. Add payment method
3. Add credits ($10 recommended)

### Issue 5: "Models not found"
**Problem**: API can't load models

**Solution**:
```bash
# Train models
python main.py --steps train
```

## 📊 Verification Checklist

After setup, verify everything is working:

```bash
# 1. Check Docker
docker ps
# ✓ Should see revenue_neo4j running

# 2. Check CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
# ✓ Should print: CUDA: True

# 3. Check API Server
curl http://localhost:8000/health
# ✓ Should return JSON with "status": "healthy"

# 4. Check Web Interface
curl http://localhost:3000
# ✓ Should return HTML

# 5. Test Endpoint
python test_endpoints.py
# ✓ Should show 200 OK for all endpoints
```

## 🔄 Daily Workflow

### Starting the Platform
```bash
# 1. Start Docker Desktop (if not running)

# 2. Start Neo4j (if not running)
docker-compose up -d neo4j

# 3. Start API Server (Terminal 1)
python -m api.server

# 4. Start Web Interface (Terminal 2)
python -m http.server 3000 --directory web

# 5. Open browser
# http://localhost:3000
```

### Stopping the Platform
```bash
# 1. Stop API Server
# Press Ctrl+C in Terminal 1

# 2. Stop Web Interface
# Press Ctrl+C in Terminal 2

# 3. Stop Neo4j (optional)
docker-compose down
```

## 📁 Project Structure

```
walmart_dataset/
├── api/                    # FastAPI server
│   ├── server.py          # Main API server
│   ├── schemas.py         # Request/response models
│   └── middleware.py      # Security middleware
├── forecasting/           # ML models
│   ├── models/           # Trained model files
│   ├── model_lstm.py     # LSTM implementation
│   ├── model_transformer.py
│   └── predict.py        # Prediction pipeline
├── rag/                   # RAG system
│   ├── rag_pipeline.py   # Main RAG logic
│   └── faiss_index/      # Vector indices
├── knowledge_graph/       # Neo4j integration
│   ├── neo4j_loader.py   # Load data to Neo4j
│   └── neo4j_query.py    # Query Neo4j
├── web/                   # Web interface
│   ├── index.html        # Main page
│   ├── styles.css        # Styling
│   └── script.js         # Frontend logic
├── data/                  # Data files
│   ├── raw/              # Original data
│   └── processed/        # Processed data
├── docker/                # Docker configs
│   └── docker-compose.yml
├── .env                   # Environment variables
├── requirements.txt       # Python dependencies
└── main.py               # Main entry point
```

## 🎓 Next Steps

1. **Explore the Web Interface**: http://localhost:3000
   - Try example commands
   - Ask custom questions
   - Get forecasts for different stores

2. **Explore Neo4j**: http://localhost:7474
   - Visualize store relationships
   - Run Cypher queries
   - Understand the graph structure

3. **Read Documentation**:
   - `README.md` - Project overview
   - `GUARDRAILS.md` - Security features
   - `NEO4J_GUIDE.md` - Graph database guide
   - `FINAL_HEALTH_REPORT.md` - System status

4. **Customize**:
   - Modify models in `forecasting/`
   - Add new endpoints in `api/server.py`
   - Customize web UI in `web/`

## 🆘 Getting Help

- **Health Check**: http://localhost:8000/health
- **Monitoring**: http://localhost:8000/monitoring/stats
- **Logs**: Check `logs/` directory
- **API Docs**: http://localhost:8000/docs

## 🎉 You're All Set!

Your Revenue Intelligence Platform is now running. Enjoy exploring your AI-powered forecasting system!

**Quick Links**:
- 🌐 Web Interface: http://localhost:3000
- 📚 API Docs: http://localhost:8000/docs
- 🔍 Neo4j Browser: http://localhost:7474
- 📊 Monitoring: http://localhost:8000/monitoring/stats
