# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains benchmark evaluation tools for MemMachine, specifically the LoCoMo (Long Conversation Memory) benchmark. It evaluates episodic memory performance through two main approaches:
- **Episodic Memory**: Direct memory search using MemMachine's query interface
- **Episodic Agent**: Agent-based approach using OpenAI Agents SDK with memory tools

## Prerequisites

- **MemMachine Backend**: Must be installed and running with `memmachine-server`
- **Configuration**: Copy your `cfg.yml` to `locomo/locomo_config.yaml` (required for both tools)
- **Environment**: Create a `.env` file with required API keys:
  - `OPENAI_API_KEY` (required)
  - `NEO4J_PASSWORD` (required for Neo4j storage backend)
  - `TRACE_API_KEY` (optional, for episodic_agent tracing)

## Project Structure

```
evaluation/
├── locomo/
│   ├── locomo10.json           # Benchmark dataset (2.8MB conversation data)
│   ├── episodic_memory/        # Direct memory search approach
│   │   ├── locomo_config.yaml  # MemMachine configuration
│   │   ├── locomo_ingest.py    # Ingest conversations into memory
│   │   ├── locomo_search.py    # Search and answer questions
│   │   ├── locomo_evaluate.py  # LLM judge evaluation
│   │   ├── generate_scores.py  # Calculate final scores
│   │   └── locomo_delete.py    # Clean up test data
│   └── episodic_agent/         # Agent-based approach
│       ├── locomo_agent.py     # Core agent implementation
│       ├── run_experiments.py  # Run experiments
│       ├── evals.py            # Evaluate results
│       └── memmachine_locomo.py # MemMachine integration
```

## Common Commands

### Episodic Memory Approach

All commands should be run from `locomo/episodic_memory/`:

```bash
cd locomo/episodic_memory

# 1. Ingest conversations (only once per test run)
python locomo_ingest.py --data-path ../locomo10.json

# 2. Search and answer questions
python locomo_search.py --data-path ../locomo10.json --target-path results.json

# 3. Evaluate responses with LLM judge
python locomo_evaluate.py --data-path results.json --target-path evaluation_metrics.json

# 4. Generate final scores
python generate_scores.py

# 5. Clean up test data
python locomo_delete.py --data-path ../locomo10.json
```

### Episodic Agent Approach

All commands should be run from `locomo/episodic_agent/`:

```bash
cd locomo/episodic_agent

# 1. Ingest conversations (same as episodic_memory)
python locomo_ingest.py --data-path ../locomo10.json

# 2. Run agent-based experiments
python run_experiments.py --method search --dataset ../locomo10.json

# 3. Evaluate results (uses git commit ID in filename)
commit_id=$(git rev-parse --short=7 HEAD)
python evals.py --input_file results_IM_$commit_id.json --output_file evaluation.json

# 4. Generate scores
python generate_scores.py --input_path evaluation.json

# 5. Clean up (same as episodic_memory)
python locomo_delete.py --data-path ../locomo10.json
```

## Architecture

### Data Flow

1. **Ingestion**: Conversations from `locomo10.json` are ingested into MemMachine
   - Each conversation has a `group_id`, multiple sessions, and two speakers
   - Messages stored as episodic memories with timestamps and metadata
   - Blip captions attached to some messages for image context

2. **Search/Query**: Questions answered by querying episodic memory
   - **Episodic Memory**: Direct query → format memories → LLM generates answer
   - **Episodic Agent**: Agent decides whether to search more or generate answer (up to 30 turns)

3. **Evaluation**: LLM judge compares generated answers to ground truth
   - Categories: 1=multi_hop, 2=temporal, 3=open_domain, 4=single_hop
   - Binary scoring: CORRECT (1) or WRONG (0)

4. **Scoring**: Aggregate accuracy by category and overall

### Key Configuration Details

The `locomo_config.yaml` specifies:
- **Storage**: Neo4j vector graph store (default localhost:7687)
- **Embedder**: OpenAI text-embedding-3-small
- **Reranker**: RRF-hybrid (combines identity, BM25, and cross-encoder)
- **Model**: gpt-4o-mini for answer generation

### Memory Episode Structure

Episodes stored with:
- `content`: Message text
- `producer`: Speaker name
- `timestamp`: Message timestamp
- `user_metadata`:
  - `source_timestamp`: Original conversation timestamp
  - `source_speaker`: Speaker name
  - `blip_caption`: Optional image description

### Agent Implementation (episodic_agent)

The `locomo_agent.py` implements an executor agent using OpenAI Agents SDK:
- Starts with pre-fetched memories from initial query
- Can call `search_conversation_session_memory` tool for more context
- Limited to 30 turns, with soft limits at 5 and 10 turns
- Must convert relative time references to absolute dates

## Output Format

Final score output:
```
Mean Scores Per Category:
          llm_score  count         type
category
1            0.8050    282    multi_hop
2            0.7259    321     temporal
3            0.6458     96  open_domain
4            0.9334    841   single_hop

Overall Mean Scores:
llm_score    0.8487
```

## Important Notes

- Category 5 questions are skipped during evaluation
- The episodic_memory approach uses simpler direct search
- The episodic_agent approach uses multi-turn agent reasoning
- Results are saved with intermediate progress (safe to interrupt)
- Both approaches use the same ingestion and deletion scripts
