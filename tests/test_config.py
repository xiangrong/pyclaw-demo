from pathlib import Path

from pyclaw.infra.config import load_config


def test_config_loads_web_search_keys(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  api_key: model-key
web_search:
  tavily_api_key: tavily-key
  brave_api_key: brave-key
""",
        encoding="utf-8",
    )

    cfg = load_config(str(config_path))

    assert cfg.web_search.tavily_api_key == "tavily-key"
    assert cfg.web_search.brave_api_key == "brave-key"
    assert cfg.max_iterations == 90
    assert cfg.effective_max_iterations == 90
    assert cfg.user_memory.enabled is True
    assert cfg.user_memory.backend == "sqlite"
    assert cfg.user_memory.external_enabled is False
    assert cfg.user_memory.auto_consolidate is True
    assert cfg.user_memory.consolidation_interval_hours == 24.0
    assert cfg.user_memory.consolidation_stale_after_days == 90
    assert cfg.document_rag.enabled is True
    assert cfg.document_rag.table_name == "document_chunks"
    assert cfg.document_rag.auto_retrieve is True
    assert cfg.document_rag.default_limit == 5


def test_config_loads_user_memory_env_overrides(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  api_key: model-key
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYCLAW_USER_MEMORY_BACKEND", "hybrid")
    monkeypatch.setenv("PYCLAW_USER_MEMORY_EXTERNAL_PROVIDER", "mem0")
    monkeypatch.setenv("PYCLAW_USER_MEMORY_EXTERNAL_ENABLED", "true")
    monkeypatch.setenv("PYCLAW_USER_MEMORY_AUTO_CONSOLIDATE", "false")
    monkeypatch.setenv("PYCLAW_USER_MEMORY_CONSOLIDATION_INTERVAL_HOURS", "12")
    monkeypatch.setenv("PYCLAW_USER_MEMORY_CONSOLIDATION_STALE_AFTER_DAYS", "45")
    monkeypatch.setenv("MEM0_API_KEY", "mem0-key")

    cfg = load_config(str(config_path))

    assert cfg.user_memory.backend == "hybrid"
    assert cfg.user_memory.external_provider == "mem0"
    assert cfg.user_memory.external_enabled is True
    assert cfg.user_memory.mem0_api_key == "mem0-key"
    assert cfg.user_memory.auto_consolidate is False
    assert cfg.user_memory.consolidation_interval_hours == 12.0
    assert cfg.user_memory.consolidation_stale_after_days == 45


def test_config_loads_document_rag_settings(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  api_key: model-key
document_rag:
  enabled: false
  db_path: /tmp/docs-db
  table_name: docs
  auto_retrieve: false
  default_limit: 7
  collection: company
  chunk_chars: 900
  chunk_overlap_chars: 100
""",
        encoding="utf-8",
    )

    cfg = load_config(str(config_path))

    assert cfg.document_rag.enabled is False
    assert cfg.document_rag.db_path == "/tmp/docs-db"
    assert cfg.document_rag.table_name == "docs"
    assert cfg.document_rag.auto_retrieve is False
    assert cfg.document_rag.default_limit == 7
    assert cfg.document_rag.collection == "company"
    assert cfg.document_rag.chunk_chars == 900
    assert cfg.document_rag.chunk_overlap_chars == 100
