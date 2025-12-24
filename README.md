# IntelliFraud Copilot

A production-grade Multi-Agent RAG System for Financial Fraud Analysis.

## 🚀 Overview

IntelliFraud Copilot is an AI-powered fraud investigation system that combines:
- **Multi-Agent Architecture**: Specialized agents for different investigation tasks
- **Hybrid RAG**: Vector + keyword search with policy documents
- **Real-time Analysis**: Transaction monitoring and anomaly detection
- **Compliance Integration**: AML/KYC policy enforcement

## 🏗️ Architecture

### System Components
┌─────────────────────────────────────────────────────────────┐
│ Frontend (Streamlit) │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ API Gateway (FastAPI) │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ Agent Orchestrator │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│ │ Planner │ │Transaction│ │ Policy │ │ Reasoning│ │
│ │ Agent │ │ Agent │ │ RAG Agent│ │ Agent │ │
│ └──────────┘ └──────────┘ └──────────┘ └──────────┘ │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ Retrieval Layer │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│ │ Vector DB│ │ SQL │ │Metadata │ │
│ │ (Qdrant) │ │ (Postgres)│ │ Filters │ │
│ └──────────┘ └──────────┘ └──────────┘ │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ LLM Layer │
│ ┌──────────┐ ┌──────────┐ │
│ │ OpenAI │ │ DeepSeek │ │
│ │ (GPT-4) │ │ (Reason) │ │
│ └──────────┘ └──────────┘ │
└─────────────────────────────────────────────────────────────┘



## 🛠️ Installation

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- PostgreSQL 15+
- Redis 7+
- Qdrant Vector Database

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/intellifraud-copilot.git
   cd intellifraud-copilot
   
2. Set up the environment variables
cp .env.example .env
# Edit .env with your API keys and configurations
3. 