# Quick Start Guide

## ⚡ 5-Minute Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
```

### 2. Start Neo4j
```bash
cd docker
docker-compose up -d neo4j
```

### 3. Configure Environment
Create `.env`:
```
OPENAI_API_KEY=your_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

### 4. Run Platform
```bash
# Terminal 1
python -m api.server

# Terminal 2
python -m http.server 3000 --directory web
```

### 5. Access
- **Web UI**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **Neo4j**: http://localhost:7474

## 🔄 Daily Use

**Start**:
```bash
docker-compose up -d neo4j
python -m api.server  # Terminal 1
python -m http.server 3000 --directory web  # Terminal 2
```

**Stop**: Press Ctrl+C in both terminals

## 📖 Full Guide
See `SETUP_GUIDE.md` for detailed instructions.
