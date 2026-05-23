# 📦 PyClaw 安装指南

欢迎使用 PyClaw！本文档将指导您如何在本地环境或服务器上安装、配置并运行 PyClaw AI Agent。

## 💻 系统要求
- **操作系统**: macOS / Linux / Ubuntu / CentOS 等
- **Python**: Python 3.9 或更高版本
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
python3 -m venv venv
source venv/bin/activate
```

*注意：如果你的系统默认 python3 版本低于 3.10，请先升级 Python 或使用 Conda。*

**3. 安装依赖与包**
```bash
pip install --upgrade pip
pip install -e .
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

## ❓ 常见问题
- **遇到 `command not found: pyclaw`**
  如果您是一键安装的，请确保 `~/.local/bin` 已经加入到了您的 `PATH` 环境变量中。
  您可以执行 `export PATH="$HOME/.local/bin:$PATH"` 然后重试。

- **检查当前版本**
  运行命令检查您安装的 PyClaw 版本：
  ```bash
  pyclaw --version
  ```on
  ```