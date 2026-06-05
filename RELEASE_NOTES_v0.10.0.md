# PyClaw v0.10.0 - Semantic Memory & Local Intelligence

This release introduces **Semantic Memory**, transforming PyClaw from a session-based assistant into a long-term partner with persistent, searchable knowledge.

## 🚀 New Features

### 🧠 Semantic Memory (LanceDB)
- **Persistent Recall**: PyClaw now automatically indexes past conversations and retrieves relevant historical context using vector search (RAG).
- **LanceDB Backend**: A lightweight, serverless vector database that stores knowledge directly in your project directory.
- **Proactive Search**: New `search_memory` tool allows the agent to explicitly search its long-term memory when needed.

### 📡 Flexible Embedding Options
- **Local Embeddings**: Support for `BAAI/bge-small-zh-v1.5` running locally via `sentence-transformers`. Zero API cost, zero latency, and 100% privacy.
- **Mixed API Providers**: Configure different endpoints for Chat (e.g., Volcengine/Ark) and Embeddings (e.g., SiliconFlow) in a single configuration.
- **Cloud Support**: Native compatibility with OpenAI and Volcengine Ark embedding models.

## 🛠️ Improvements & Fixes
- **Async Prompting**: Dynamic system prompt generation is now fully asynchronous, enabling real-time memory retrieval without blocking.
- **Dependency Management**: Updated `requirements.txt` and `pyproject.toml` with `lancedb`, `pyarrow`, and `sentence-transformers`.
- **Bug Fixes**: Resolved `asyncio` import issues and improved multi-process signal handling during restarts.

## 📦 Installation
```bash
pip install pyclaw==0.10.0
```
*Note: Local embedding support requires `pip install sentence-transformers`.*

---
**Full Changelog**: [v0.9.0...v0.10.0](https://github.com/xiangrong/pyclaw-demo/compare/v0.9.0...v0.10.0)
