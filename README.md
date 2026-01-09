---
title: Multi Game Arena
sdk: docker
app_port: 7860
---

# 🎮 Multi Game Arena

# AI-Powered Gaming Platform

A full-stack gaming platform featuring real-time personality analysis through machine learning. Players engage in three distinct game modes while an ML system analyzes their behavior patterns to create personalized AI opponents.

**Live Demo:** [https://multi-game-arena.vercel.app/] 

---

## Project Overview

This project demonstrates practical implementation of:
- **Real-time ML personality classification** (8 archetypes, 75%+ validation accuracy)
- **Full-stack architecture** (React + FastAPI + PostgreSQL)
- **Production CI/CD pipeline** (GitHub Actions → Docker → Hugging Face Spaces)
- **3D game development** (Three.js/React Three Fiber with physics)

### What Makes This Unique

Instead of static difficulty settings, the AI opponent adapts based on the player's actual behavior. A Random Forest classifier analyzes gameplay actions (aggression, patience, risk-taking) to predict personality type, then adjusts AI strategy accordingly.

---

## Technical Stack

### Backend
- **Framework:** FastAPI (Python 3.11)
- **ML Pipeline:** Scikit-learn (Random Forest, SVM ensemble)
- **Database:** PostgreSQL + SQLAlchemy ORM
- **Task Orchestration:** Prefect (ML retraining workflows)
- **Deployment:** Docker, Hugging Face Spaces

### Frontend
- **Framework:** React 18 + TypeScript + Vite
- **3D Engine:** Three.js via React Three Fiber
- **Physics:** Rapier (WebAssembly physics engine)
- **State Management:** TanStack Query
- **Styling:** Tailwind CSS + Shadcn UI

### DevOps
- **CI/CD:** GitHub Actions (6-stage pipeline)
- **Monitoring:** Prometheus + Grafana
- **Containerization:** Docker Compose (multi-service orchestration)

---

## Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│                 │         │                  │         │                 │
│  React Frontend │◄───────►│  FastAPI Backend │◄───────►│   PostgreSQL    │
│  (Vercel)       │  HTTP   │  (HF Spaces)     │  SQL    │   (Actions DB)  │
│                 │         │                  │         │                 │
└─────────────────┘         └──────────────────┘         └─────────────────┘
                                     │
                                     │ Inference
                                     ▼
                            ┌──────────────────┐
                            │  ML Classifier   │
                            │  (1.4GB model)   │
                            │  Random Forest   │
                            └──────────────────┘
```
### Current Production Status

**Available in Demo:**
- ✅ Fighting Game (full ML integration)

**In Development:**
- ⏸️ Badminton Mode (performance optimization in progress)
- ⏸️ Racing Mode (WebGL stability improvements)

The ML personality classifier and AI adaptation system are fully functional across all game modes in local development. Production deployment currently focuses on the Fighting Game to ensure optimal performance across all devices.
### Data Flow

1. **Player Action** → Frontend captures input (punch, racket swing, steering)
2. **Feature Extraction** → Backend converts actions to ML features (aggression rate, combo preference, etc.)
3. **Personality Prediction** → Model classifies player into 1 of 8 archetypes
4. **AI Adaptation** → Opponent adjusts strategy based on prediction
5. **Database Storage** → All actions logged for continuous model improvement

---

## Machine Learning Implementation

### Feature Engineering

Each game mode extracts 4 behavioral features:

**Fighting Game:**
- Aggression Rate (attack actions / total actions)
- Defense Ratio (block actions / total actions)  
- Combo Preference (average combo length)
- Reaction Time (success rate proxy)

**Badminton:**
- Shot Variety (unique shot types used)
- Power Control (variance in shot power)
- Court Positioning (movement efficiency)
- Rally Patience (average rally length)

**Racing:**
- Speed Preference (average velocity)
- Precision Level (1 - crash rate)
- Overtaking Aggression (overtake attempts / total)
- Consistency (speed variance)

### Model Architecture

```python
# Hybrid Classification System
class CrossGamePersonalityClassifier:
    - Game-specific regressors (3x Random Forest)
    - Meta-regressor (ensemble predictions)
    - Personality classifier (VotingClassifier: RF + ExtraTrees + GradientBoosting)
    - Playstyle classifier (RandomForest)
```

**Training Results:**
- Personality classification: **77% validation accuracy**
- Cross-game consistency: **0.82 R² score**
- Training set: 10,000 synthetic samples with realistic distributions
- Validation gap: <10% (minimal overfitting)

### Personality Archetypes

1. **🔥 Aggressive Dominator** - High aggression, low patience
2. **🧠 Strategic Analyst** - High analytical thinking, precise timing
3. **⚡ Risk-Taking Maverick** - High risk tolerance, unpredictable
4. **🛡️ Defensive Tactician** - High patience, defensive play
5. **🎯 Precision Master** - High precision focus, controlled actions
6. **🌪️ Chaos Creator** - Maximum aggression + risk, minimum precision
7. **📊 Data-Driven Player** - High analytical thinking, low risk
8. **🏆 Victory Seeker** - High competitive drive, balanced traits

---

## CI/CD Pipeline

GitHub Actions workflow with 6 automated stages:

```yaml
1. Code Quality Checks
   ├─ Black (formatting)
   ├─ Flake8 (linting)
   └─ isort (import sorting)

2. Unit Tests + Coverage
   ├─ pytest (85% coverage)
   └─ PostgreSQL test database

3. Data Validation
   ├─ DeepChecks (data integrity)
   └─ Model performance tests

4. Docker Build
   ├─ Multi-stage build
   └─ Push to Docker Hub

5. ML Pipeline Execution
   ├─ Prefect orchestration
   └─ Model retraining checks

6. Deployment
   └─ Sync to Hugging Face Spaces
```

**Notifications:** Discord webhooks for pipeline status

---

## Key Implementation Challenges

### 1. Real-Time ML Inference
**Challenge:** Sub-300ms response time for AI decisions  
**Solution:** 
- Pre-loaded model (1.4GB) in memory
- Feature caching with 2-second TTL
- Async request handling with FastAPI

### 2. WebGL Context Management
**Challenge:** Browser crashes on intensive 3D rendering  
**Solution:**
- Implemented context loss recovery handlers
- Reduced polygon count in racing track (20 elements vs 100)
- Disabled anti-aliasing, enabled `powerPreference: "high-performance"`

### 3. Concurrent Database Access
**Challenge:** Race conditions during personality updates  
**Solution:**
- SQLAlchemy connection pooling (10 connections, 20 overflow)
- Optimistic locking with `onupdate` timestamps
- Retry logic with exponential backoff

### 4. Model Overfitting
**Challenge:** 95% training accuracy, 60% validation  
**Solution:**
- Added regularization (min_samples_leaf=5, max_leaf_nodes=50)
- Reduced feature count (36 → 21 engineered features)
- Cross-validation during training (5-fold CV)
- Result: Training 77%, Validation 75% (acceptable gap)

---

## Running Locally

### Prerequisites
- Docker & Docker Compose
- Node.js 18+
- Python 3.11+

### Quick Start

```bash
# Clone repository
git clone https://github.com/mominazahid10/multi_game-arena.git
cd multi_game-arena

# Set environment variables
cp .env.example .env
# Edit .env with your DB_PASSWORD and SECRET_KEY

# Download ML model
python backend/download_model.py

# Start all services
docker-compose up --build

# Access application
# Frontend: http://localhost:5173
# Backend API: http://localhost:8000/docs
# Grafana: http://localhost:3000
```

### Development Mode

```bash
# Backend only
cd backend
pip install -r requirements.txt
uvicorn main_optimized:app --reload --port 8000

# Frontend only  
cd frontend
npm install
npm run dev
```

---

## Project Metrics

- **Total Code:** ~15,000 lines (TypeScript + Python)
- **Model Size:** 1.4GB (trained on 10,000 samples)
- **API Response Time:** <250ms average
- **Database:** 7 tables, normalized schema
- **Test Coverage:** 85% (pytest)
- **Docker Image:** 3GB (optimized multi-stage build)

---

## Future Enhancements

- [ ] Implement ONNX Runtime for faster inference
- [ ] Add multiplayer via WebRTC
- [ ] Integrate LLM for natural language AI taunts
- [ ] Deploy frontend to Cloudflare Workers
- [ ] Add reinforcement learning for adaptive AI

---

## Learning Outcomes

This project helped me develop:
- **Production ML:** Model training, validation, deployment, monitoring
- **Full-Stack Skills:** REST API design, database optimization, frontend state management
- **DevOps:** Docker orchestration, CI/CD pipelines, automated testing
- **3D Programming:** WebGL optimization, physics simulation, real-time rendering
- **System Design:** Microservices, caching strategies, error handling

---

## Contact

**Momina Zahid**  
Developer | Machine Learning Enthusiast

📧 [Email](mailto:mominazd12@example.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/mominazahidd)  
💻 [GitHub](https://github.com/mominazahid10)

---


