---
title: Multi Game Arena
sdk: docker
app_port: 7860
---

# 🎮 Multi Game Arena

Multi Game Arena is an AI-powered system that analyzes player behavior across multiple games
and predicts personality archetypes and playstyles using machine learning.

## 🚀 Features
- Cross-game personality prediction
- Hybrid ML models (classification + regression)
- Automated ML monitoring pipeline
- Docker-based deployment
- Memory-safe inference (single-core execution)
- Discord notifications for pipeline status

## 🐳 Docker Setup
This Space runs using **Docker**. Hugging Face will automatically:
- Build the Docker image using your `Dockerfile`
- Expose the application on port **7860**

Make sure your Dockerfile includes:
```dockerfile
EXPOSE 7860
