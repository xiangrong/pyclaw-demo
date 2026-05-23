#!/bin/bash
# PyClaw 启动脚本

cd "$(dirname "$0")"

echo "🦞 PyClaw - Python AI Agent"
echo "============================"

# 检查并激活虚拟环境
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 检查依赖
if ! python3 -c "import mcp" 2>/dev/null; then
    echo "📦 安装依赖 (MCP 需要 Python 3.10+)..."
    python3 -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
fi

# 检查配置文件
CONFIG_FILE="$HOME/.config/pyclaw/config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "⚙️ 创建配置文件..."
    python3 -m pyclaw init
    echo ""
    echo "✏️ 请编辑配置文件: $CONFIG_FILE"
    echo "填入你的 Telegram Token, API Key 和 Amap Key"
    echo ""
    echo "然后重新运行: ./start.sh"
    exit 0
fi

echo "🚀 启动中..."
echo "📝 配置文件: $CONFIG_FILE"
echo ""

python3 -m pyclaw start
