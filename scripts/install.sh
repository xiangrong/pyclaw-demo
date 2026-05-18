#!/bin/bash
# ============================================================================
# PyClaw 一键安装脚本
# ============================================================================
# 使用方式:
#   curl -fsSL https://raw.githubusercontent.com/xiangrong/pyclaw-demo/main/scripts/install.sh | bash
#
# 或者本地测试:
#   bash scripts/install.sh
# ============================================================================

set -e

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'
BOLD='\033[1m'

# 配置
REPO_URL="https://github.com/xiangrong/pyclaw-demo"
INSTALL_DIR="$HOME/.pyclaw"
VENV_DIR="$INSTALL_DIR/venv"
BIN_DIR="$HOME/.local/bin"

# ============================================================================
# 辅助函数
# ============================================================================

print_banner() {
    echo ""
    echo -e "${BLUE}${BOLD}"
    echo "╔══════════════════════════════════════════════════╗"
    echo "║           🐍 PyClaw AI Agent 安装程序           ║"
    echo "╚══════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

log_info() {
    echo -e "${BLUE}→${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

# ============================================================================
# 系统检查
# ============================================================================

# 根据 CPU 架构选择正确的执行方式
arch_python() {
    if [[ "$(uname -m)" == "arm64" ]]; then
        # Apple Silicon - 强制 arm64 架构执行
        arch -arm64 python3 "$@"
    else
        # Intel 或其他架构
        python3 "$@"
    fi
}

arch_pip() {
    if [[ "$(uname -m)" == "arm64" ]]; then
        # Apple Silicon - 强制 arm64 架构执行
        arch -arm64 "$VENV_DIR/bin/pip" "$@"
    else
        # Intel 或其他架构
        "$VENV_DIR/bin/pip" "$@"
    fi
}

check_system() {
    log_info "检查系统环境..."

    # 检查架构
    CPU_ARCH=$(uname -m)
    PYTHON_ARCH=$(python3 -c "import platform; print(platform.machine())")
    
    if [[ "$CPU_ARCH" == "arm64" ]]; then
        log_success "CPU: Apple Silicon (arm64)"
        if [[ "$PYTHON_ARCH" == "arm64" ]]; then
            log_success "Python: 原生 arm64"
        else
            log_warn "Python 运行在 Rosetta 转译模式下，将强制 arm64 架构安装"
        fi
    else
        log_success "CPU: Intel (x86_64)"
    fi

    # 检查 Python
    if ! command -v python3 &> /dev/null; then
        log_error "未找到 python3，请先安装 Python 3.9+"
        exit 1
    fi

    PYTHON_VERSION=$(arch_python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log_success "Python $PYTHON_VERSION 已找到"

    # 检查 Git
    if ! command -v git &> /dev/null; then
        log_error "未找到 git，请先安装 Git"
        exit 1
    fi
    log_success "Git 已找到"

    # 检查 pip
    if ! python3 -m pip --version &> /dev/null; then
        log_error "pip 不可用，请先安装 pip"
        exit 1
    fi
    log_success "pip 已找到"
}

# ============================================================================
# 安装
# ============================================================================

install_pyclaw() {
    log_info "创建安装目录: $INSTALL_DIR"
    mkdir -p "$INSTALL_DIR"

    # Clone 或更新代码
    if [ -d "$INSTALL_DIR/pyclaw-demo/.git" ]; then
        log_info "更新现有代码..."
        cd "$INSTALL_DIR/pyclaw-demo"
        git pull origin main
    else
        log_info "克隆代码仓库..."
        cd "$INSTALL_DIR"
        git clone "$REPO_URL" pyclaw-demo
    fi

    # 创建虚拟环境（注意：venv 创建不能用 arch -arm64 前缀，直接创建）
    if [ ! -d "$VENV_DIR" ]; then
        log_info "创建虚拟环境..."
        python3 -m venv "$VENV_DIR"
    fi

    # 安装依赖（使用架构适配的 pip）
    log_info "安装依赖..."
    arch_pip install --upgrade pip
    arch_pip install -e "$INSTALL_DIR/pyclaw-demo"

    # 创建可执行文件链接
    log_info "创建命令链接..."
    mkdir -p "$BIN_DIR"
    
    # 包装脚本（Apple Silicon 下强制 arm64 执行）
    cat > "$BIN_DIR/pyclaw" << EOF
#!/bin/bash
if [[ "\$(uname -m)" == "arm64" ]]; then
    exec arch -arm64 "$VENV_DIR/bin/pyclaw" "\$@"
else
    exec "$VENV_DIR/bin/pyclaw" "\$@"
fi
EOF
    chmod +x "$BIN_DIR/pyclaw"

    log_success "PyClaw 安装完成！"
}

# ============================================================================
# PATH 检查
# ============================================================================

check_path() {
    if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
        log_warn "$BIN_DIR 不在你的 PATH 中"
        echo ""
        echo "请手动添加到你的 shell 配置文件:"
        echo "  export PATH=\"$BIN_DIR:\$PATH\""
        echo ""
        echo "或者临时使用:"
        echo "  $BIN_DIR/pyclaw --help"
    fi
}

# ============================================================================
# 完成
# ============================================================================

print_finish() {
    echo ""
    echo -e "${GREEN}═════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${GREEN}🎉 安装成功！${NC}"
    echo ""
    echo "下一步:"
    echo "  1. 初始化配置: pyclaw init"
    echo "  2. 编辑配置文件: ~/.pyclaw/config.yaml"
    echo "  3. 启动服务: pyclaw start"
    echo ""
    echo "常用命令:"
    echo "  pyclaw --help     - 查看帮助"
    echo "  pyclaw init       - 创建配置文件"
    echo "  pyclaw start      - 启动 Agent 服务"
    echo ""
    echo -e "${GREEN}═════════════════════════════════════════════════════════${NC}"
    echo ""
}

# ============================================================================
# 主流程
# ============================================================================

main() {
    print_banner
    check_system
    echo ""
    install_pyclaw
    echo ""
    check_path
    print_finish
}

main "$@"
