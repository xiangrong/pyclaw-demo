# 📦 PyClaw 安装指南

欢迎使用 PyClaw！本文档将指导您如何在本地环境或服务器上安装、配置并运行 PyClaw AI Agent。

## 💻 系统要求
- **操作系统**: macOS / Linux / Ubuntu / CentOS 等
- **Python**: Python 3.10 或更高版本
- **Git**: 必须安装 Git (用于拉取代码和安装技能)

---

## 🚀 方式一：推荐安装（一键脚本）
我们提供了一个自动化的安装脚本，它会自动处理依赖、创建虚拟环境、适配 Apple Silicon (M1/M2/M3) 架构，并将 `pyclaw` 命令添加到您的环境变量中。

打开终端 (Terminal) 并运行：

```bash
curl -fsSL https://raw.githubusercontent.com/xiangrong/pyclaw-demo/main/scripts/install.sh | bash
```

*安装成功后，脚本会提示 `PyClaw 安装完成！`*

---

## 🛠️ 方式二：手动源码安装 (适合开发者)
如果您希望修改源码或参与贡献，可以使用手动安装方式。

**1. 克隆代码仓库**
```bash
git clone https://github.com/xiangrong/pyclaw-demo.git
cd pyclaw-demo
```

**2. 创建虚拟环境 (强烈推荐，确保使用 Python 3.10+)**
```bash
python3.10 -m venv venv
source venv/bin/activate
```

*注意：PyClaw 的 `pyproject.toml` 要求 Python >= 3.10。如果你的系统默认 `python3` 版本低于 3.10（例如 macOS 自带的 3.9.6），请先升级 Python，或使用 Homebrew/Conda/pyenv 安装 3.10+ 后再创建虚拟环境。*

如果已安装 Python 3.10+，但命令名不是 `python3.10`，可以把上面的命令替换成你的实际路径，例如：
```bash
/opt/homebrew/bin/python3.11 -m venv venv
```

**3. 安装依赖与包**
```bash
pip install --upgrade pip
pip install -e .
```

如果你使用 `--no-build-isolation` 或旧版安装脚本，请先确保 editable 构建依赖已安装：
```bash
pip install --upgrade pip hatchling editables
pip install --no-build-isolation -e .
```

*注意：核心依赖不包含向量数据库。如果您需要启用「语义记忆 (RAG)」功能，请额外安装：*
```bash
pip install -e ".[rag]"
```

如果还需要本地 Embedding（`embedding_base_url: "local"`），请安装完整增强依赖：
```bash
pip install -e ".[all]"
```

---

## ⚙️ 安装后的配置步骤

无论您使用哪种安装方式，成功后都需要进行如下配置：

**1. 初始化配置模板**
```bash
pyclaw init
```
*这将在 `~/.config/pyclaw/config.yaml` 生成一个配置文件。*

**2. 编辑配置文件**
填入您的 LLM API Key 以及 Telegram / 飞书的 Bot Token：
```bash
vim ~/.config/pyclaw/config.yaml
```

**3. 启动 Agent**
配置完成后，直接启动：
```bash
pyclaw start
```
如果看到 `🚀 PyClaw Agent 已启动`，就可以在 Telegram 或飞书里和它聊天了！

---

## 🔄 如何更新

如果您使用的是一键安装脚本，可以直接再次运行该脚本来拉取最新代码并更新依赖：
```bash
curl -fsSL https://raw.githubusercontent.com/xiangrong/pyclaw-demo/main/scripts/install.sh | bash
```

如果是手动安装，请进入项目目录执行：
```bash
git pull origin main
pip install -e .
```

如果一键安装脚本检测到当前 `python3` 低于 3.10，可以通过 `PYCLAW_PYTHON` 指定解释器：
```bash
PYCLAW_PYTHON=/opt/homebrew/bin/python3.11 bash scripts/install.sh
```

## ❓ 常见问题
- **遇到 `command not found: pyclaw`**
  如果您是一键安装的，请确保 `~/.local/bin` 已经加入到了您的 `PATH` 环境变量中。
  您可以执行 `export PATH="$HOME/.local/bin:$PATH"` 然后重试。

- **遇到 `Package pyclaw requires a different Python: 3.9.6 not in >=3.10`**
  说明当前安装用的是 Python 3.9.6，但 PyClaw 要求 Python >= 3.10。请安装 Python 3.10+ 后重试：
  ```bash
  brew install python@3.11
  PYCLAW_PYTHON=/opt/homebrew/bin/python3.11 bash scripts/install.sh
  ```
  如果你之前已经用 3.9 创建过 `~/.pyclaw/venv`，新版安装脚本会自动检测并重建该虚拟环境。

- **安装卡在 `Preparing metadata (pyproject.toml)` 很久**
  这通常是 pip 在构建隔离环境、解析依赖或下载较重依赖。新版安装脚本会预装 `hatchling` / `editables` 并使用 `--no-build-isolation` 安装核心包，默认安装也不再包含 `lancedb` / `pyarrow` / `tantivy` / `sentence-transformers` 等 RAG 重依赖。若需要语义记忆，再单独运行：
  ```bash
  pip install -e ".[rag]"
  ```
  如果需要本地 Embedding，再运行：
  ```bash
  pip install -e ".[all]"
  ```

- **遇到 `ModuleNotFoundError: No module named 'editables'`**
  这是 editable 安装所需的构建依赖缺失。请更新代码后重新运行一键安装脚本；如果是手动安装，请执行：
  ```bash
  pip install --upgrade pip hatchling editables
  pip install --no-build-isolation -e .
  ```

- **检查当前版本**
  运行命令检查您安装的 PyClaw 版本：
  ```bash
  pyclaw --version
  ```
