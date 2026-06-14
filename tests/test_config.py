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
