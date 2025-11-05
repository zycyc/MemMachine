# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MemMachine is an open-source memory layer for AI agents that provides three types of memory:
- **Episodic Memory**: Short-term (session) and long-term (declarative) conversational context
- **Profile Memory**: Long-term user facts and preferences stored in PostgreSQL with pgvector
- **Session Memory**: Working memory for active conversations

The system uses a graph database (Neo4j) for episodic memory and a relational database (PostgreSQL with pgvector) for profile memory.

## Development Setup

### Installation

Using uv (recommended):
```bash
uv sync              # Install core dependencies
uv sync --all-extras # Include GPU dependencies (sentence-transformers)
```

Using pip:
```bash
pip install -e "."      # Core dependencies only
pip install -e ".[gpu]" # Include GPU dependencies
```

### Running the Server

Run the FastAPI server directly:
```bash
uv run memmachine-server
# Or if not using uv:
python -m memmachine.server.app
```

The server requires:
- A YAML configuration file (see `sample_configs/episodic_memory_config.cpu.sample`)
- Environment variables (see `sample_configs/server_config.sample`)
- Running Neo4j and PostgreSQL instances (use `docker-compose.yml` for local dev)

### Docker Compose

Start all services (MemMachine, PostgreSQL, Neo4j):
```bash
docker-compose up
```

Or use the helper script:
```bash
./memmachine-compose.sh up
```

### NLTK Setup

The project requires NLTK data packages:
```bash
uv run memmachine-nltk-setup
```

## Commands

### Linting and Formatting

Run Ruff linter:
```bash
uv run ruff check
```

Run Ruff formatter:
```bash
uv run ruff format
```

Run Mypy type checker:
```bash
uv run mypy src
```

### Testing

Run all unit tests:
```bash
uv run pytest
# Or: pytest
```

Run specific test file:
```bash
pytest tests/path/to/test_file.py
```

Run integration tests (requires Docker):
```bash
pytest --integration
```

Integration tests use testcontainers to spin up Neo4j and PostgreSQL databases.

### Database Schema Sync

Sync the profile memory database schema:
```bash
uv run memmachine-sync-profile-schema
```

## Architecture

### Core Components

**Episodic Memory** (`src/memmachine/episodic_memory/`):
- `episodic_memory.py`: Orchestrates short-term and long-term memory for a session
- `episodic_memory_manager.py`: Manages lifecycle of memory instances across sessions
- `short_term_memory/`: Session-based working memory implementation
- `long_term_memory/`: Persistent declarative memory using graph storage
- `declarative_memory/`: Complex subsystem for memory encoding with derivative derivers, mutators, and related episode postulators

**Profile Memory** (`src/memmachine/profile_memory/`):
- `profile_memory.py`: User profile management with LLM-powered extraction
- `storage/`: PostgreSQL storage layer with pgvector for semantic search
- Uses `ProfileUpdateTracker` to batch profile updates based on message count and time intervals

**Server** (`src/memmachine/server/`):
- `app.py`: FastAPI application with RESTful endpoints and FastMCP integration
- Exposes memory operations as both HTTP APIs and MCP tools for LLMs
- Uses Prometheus for metrics collection

**Common** (`src/memmachine/common/`):
- `embedder/`: Abstraction for embedding models (OpenAI, AWS Bedrock, Ollama)
- `language_model/`: LLM integration layer
- `vector_graph_store/`: Neo4j graph database interface
- `reranker/`: Hybrid search reranking (BM25, identity, RRF)
- `builder.py`: Factory pattern for component construction

### Memory Context

Each memory instance is uniquely identified by `MemoryContext`:
- `group_id`: Group or shared context identifier
- `session_id`: Individual conversation session
- `agent_ids`: List of AI agents in the session
- `user_ids`: List of users in the session

### Configuration System

YAML configuration drives component assembly:
- Models: OpenAI, AWS Bedrock, or OpenAI-compatible (Ollama)
- Embedders: Vector embedding providers
- Rerankers: Search result reranking strategies
- Storage: Database connection details
- Memory behavior: Message capacity, token limits, derivative derivation

## Code Style

- Formatter: Ruff (enforced in CI)
- Linter: Ruff with rules E, F, I, W (ignores E501 line length)
- Type checker: Mypy with Pydantic plugin
- Python 3.12+ required
- All commits must be signed with `-sS` flags

## Testing Strategy

- Unit tests: Mock external dependencies (databases, APIs)
- Integration tests: Use testcontainers for real database instances
- Mark integration tests with `@pytest.mark.integration`
- Async fixtures use session scope by default

## API Interfaces

The server provides:
1. **REST API**: FastAPI endpoints for memory operations
2. **MCP Server**: FastMCP tools for LLM integration
3. **Prometheus Metrics**: `/metrics` endpoint for observability

## Key Patterns

**Builder Pattern**: Used extensively for constructing embedders, language models, rerankers, and derivative components from configuration.

**Reference Counting**: `EpisodicMemory` instances use reference counting for lifecycle management within `EpisodicMemoryManager`.

**Async Operations**: Most memory operations are async, with dedicated async wrappers (`AsyncEpisodicMemory`, `AsyncPgProfileStorage`).

**Derivative System**: Long-term memory uses a pipeline:
1. Derivative Deriver: Transforms episodes (identity, sentence splitting, concatenation)
2. Derivative Mutator: Enhances derivatives (metadata, LLM processing)
3. Related Episode Postulator: Links related episodes in the graph

## Important Notes

- The source code is in `src/memmachine/`, not `memmachine/` at the root
- Configuration samples are in `sample_configs/` directory
- NLTK data must be downloaded before first use
- Docker Compose handles all infrastructure dependencies for local development
