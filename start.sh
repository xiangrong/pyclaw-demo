#!/bin/bash
# PyClaw 启动脚本

cd "$(dirname "$0")"

echo "🦞 PyClaw - Python AI Agent"
echo "============================"

# 检查依赖
if ! python3 -c "import telegram, openai, pydantic, yaml, typer" 2>/dev/null; then
    echo "📦 安装依赖..."
    pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

# 检查配置文件
CONFIG_FILE="$HOME/.config/pyclaw/config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "⚙️ 创建配置文件..."
    python3 -m pyclaw init
    echo ""
    echo "✏️ 请编辑配置文件: $CONFIG_FILE"
    echo "填入你的 Telegram Token 和 API Key"
    echo ""
    echo "然后重新运行: ./start.sh"
    exit 0
fi

echo "🚀 启动中..."
echo "📝 配置文件: $CONFIG_FILE"
echo ""

python3 -m pyclaw start
