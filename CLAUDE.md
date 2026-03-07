# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAGFlow is an open-source RAG (Retrieval-Augmented Generation) engine based on deep document understanding. It's a full-stack application with:
- Python backend (Flask-based API server)
- React/TypeScript frontend (built with Vite)
- Microservices architecture with Docker deployment
- Multiple data stores (MySQL, Elasticsearch/Infinity, Redis, MinIO)

## Architecture

### Backend (`/api/`)
- **Main Server**: `api/ragflow_server.py` - Flask application entry point
- **Apps**: Modular Flask blueprints in `api/apps/` for different functionalities:
  - `kb_app.py` - Knowledge base management
  - `dialog_app.py` - Chat/conversation handling
  - `document_app.py` - Document processing
  - `canvas_app.py` - Agent workflow canvas
  - `file_app.py` - File upload/management
- **Services**: Business logic in `api/db/services/`
- **Models**: Database models in `api/db/db_models.py`

### Core Processing (`/rag/`)
- **Document Processing**: `deepdoc/` - PDF parsing, OCR, layout analysis
- **LLM Integration**: `rag/llm/` - Model abstractions for chat, embedding, reranking
- **RAG Pipeline**: `rag/flow/` - Chunking, parsing, tokenization
- **Graph RAG**: `rag/graphrag/` - Knowledge graph construction and querying

### Agent System (`/agent/`)
- **Components**: Modular workflow components (LLM, retrieval, categorize, etc.)
- **Templates**: Pre-built agent workflows in `agent/templates/`
- **Tools**: External API integrations (Tavily, Wikipedia, SQL execution, etc.)

### Frontend (`/web/`)
- React/TypeScript with Vite framework
- Ant Design + shadcn/ui components
- State management with Zustand
- Tailwind CSS for styling

## Common Development Commands

### Backend Development
```bash
# Install Python dependencies (requires Python 3.12)
uv sync --python 3.12 --all-extras
uv run download_deps.py
pre-commit install

# Start dependent services (MySQL, Elasticsearch/Infinity, Redis, MinIO)
docker compose -f docker/docker-compose-base.yml up -d

# Add service hostnames to /etc/hosts (required for source development)
# 127.0.0.1  es01 infinity mysql minio redis sandbox-executor-manager

# Run backend (requires services to be running)
source .venv/bin/activate
export PYTHONPATH=$(pwd)
bash docker/launch_backend_service.sh

# Run tests
uv run pytest                    # All tests
uv run pytest -m p1              # High priority tests only
uv run pytest -m p2              # Medium priority tests
uv run pytest --cov              # With coverage report
uv run pytest test/unit_test/    # Unit tests only

# Linting and formatting
ruff check                       # Check for issues
ruff format                      # Format code

# Stop backend services
pkill -f "ragflow_server.py|task_executor.py"
```

### Frontend Development
```bash
cd web
npm install
npm run dev        # Development server
npm run build      # Production build
npm run lint       # ESLint
npm run test       # Jest tests
```

### Docker Development
```bash
# Full stack with Docker
cd docker
docker compose -f docker-compose.yml up -d

# Check server status
docker logs -f ragflow-server

# Stop and remove containers (WARNING: -v deletes volumes and data)
docker compose -f docker-compose.yml down
docker compose -f docker-compose.yml down -v  # Also remove volumes

# Rebuild images
docker build --platform linux/amd64 -f Dockerfile -t infiniflow/ragflow:nightly .

# Build with proxy (if behind firewall)
docker build --platform linux/amd64 \
  --build-arg http_proxy=http://YOUR_PROXY:PORT \
  --build-arg https_proxy=http://YOUR_PROXY:PORT \
  -f Dockerfile -t infiniflow/ragflow:nightly .
```

## Key Configuration Files

- `docker/.env` - Environment variables for Docker deployment
- `docker/service_conf.yaml.template` - Backend service configuration
- `pyproject.toml` - Python dependencies and project configuration
- `web/package.json` - Frontend dependencies and scripts

## Testing

- **Python**: pytest with markers (p1/p2/p3 priority levels)
  - Run all tests: `uv run pytest`
  - Run specific priority: `uv run pytest -m p1`
  - Run with coverage: `uv run pytest --cov`
  - Run specific test file: `uv run pytest test/unit_test/common/test_string_utils.py`
  - Run with parallel execution: `uv run pytest -n auto`
- **Frontend**: Jest with React Testing Library
  - Run tests: `cd web && npm run test`
- **API Tests**: HTTP API and SDK tests in `test/testcases/`
  - HTTP API tests: `test/testcases/test_http_api/`
  - SDK tests: `test/testcases/test_sdk_api/`
  - Web API tests: `test/testcases/test_web_api/`
- **Benchmark**: Performance testing in `test/benchmark/`

## Database Engines

RAGFlow supports switching between Elasticsearch (default) and Infinity:
- Set `DOC_ENGINE=infinity` in `docker/.env` to use Infinity
- Requires container restart with volume cleanup: `docker compose down -v && docker compose up -d`
- **WARNING**: Using `-v` flag deletes all existing data in volumes

## Code Style and Conventions

- **Python**:
  - Line length: 200 characters (configured in `pyproject.toml`)
  - Linter: ruff with ASYNC checks enabled
  - Formatter: ruff format
  - Pre-commit hooks enforce formatting
- **Frontend**:
  - TypeScript with strict mode
  - ESLint for linting
  - Prettier for formatting (configured with organize-imports plugin)
  - Lint-staged runs on commit

## Development Environment Requirements

- Python 3.10-3.12
- Node.js >=18.20.4
- Docker & Docker Compose
- uv package manager
- 16GB+ RAM, 50GB+ disk space

## Important Notes

- **Python Version**: Requires Python 3.12 for development (specified in pyproject.toml: `>=3.12,<3.15`)
- **Environment Setup**: Must add service hostnames to `/etc/hosts` when running from source:
  ```
  127.0.0.1  es01 infinity mysql minio redis sandbox-executor-manager
  ```
- **Proxy Settings**: Backend startup script (`docker/launch_backend_service.sh`) automatically unsets HTTP proxies
- **HuggingFace Access**: Set `HF_ENDPOINT=https://hf-mirror.com` if you cannot access HuggingFace directly
- **Package Manager**: Uses `uv` for Python dependency management (faster than pip)
- **Pre-commit Hooks**: Install with `pre-commit install` to run linting before commits
- **Backend Process Management**: Backend runs two types of processes:
  - `ragflow_server.py`: Main Flask API server
  - `task_executor.py`: Background task workers (number controlled by `WS` env var, defaults to 1)
- **Docker Images**: All official images are built for x86_64 platforms. ARM64 users must build custom images.
