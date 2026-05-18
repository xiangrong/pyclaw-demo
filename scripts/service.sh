#!/bin/bash
# ============================================================================
# PyClaw 服务管理脚本
# ============================================================================
# 使用方式:
#   bash scripts/service.sh install   - 安装为 launchd 服务
#   bash scripts/service.sh start     - 启动服务
#   bash scripts/service.sh stop      - 停止服务
#   bash scripts/service.sh restart   - 重启服务
#   bash scripts/service.sh status    - 查看服务状态
#   bash scripts/service.sh logs      - 查看日志
#   bash scripts/service.sh uninstall - 卸载服务
# ============================================================================

set -e

# 配置
PLIST_NAME="com.pyclaw.agent"
PLIST_DIR="$HOME/Library/LaunchAgents"
PLIST_PATH="$PLIST_DIR/$PLIST_NAME.plist"
PYCLAW_HOME="$HOME/.pyclaw"
PYCLAW_BIN="$HOME/.local/bin/pyclaw"

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# 检查是否已安装
check_installed() {
    if [ ! -f "$PLIST_PATH" ]; then
        log_error "服务未安装，请先运行: $0 install"
        exit 1
    fi
}

# 安装服务
cmd_install() {
    log_info "安装 PyClaw 服务..."

    if [ ! -f "$PYCLAW_BIN" ]; then
        log_error "未找到 pyclaw 命令，请先安装 PyClaw"
        exit 1
    fi

    # 创建日志目录
    mkdir -p "$PYCLAW_HOME"

    # 生成 plist 文件（替换用户名）
    mkdir -p "$PLIST_DIR"
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    TEMPLATE_PLIST="$SCRIPT_DIR/pyclaw.plist"
    
    if [ ! -f "$TEMPLATE_PLIST" ]; then
        log_error "未找到 pyclaw.plist 模板"
        exit 1
    fi

    # 替换占位符
    sed "s|REPLACE_WITH_YOUR_USER|$USER|g" "$TEMPLATE_PLIST" > "$PLIST_PATH"

    # 加载服务
    launchctl load "$PLIST_PATH"
    
    log_success "服务已安装！"
    echo ""
    echo "服务信息:"
    echo "  • Plist: $PLIST_PATH"
    echo "  • 日志: $PYCLAW_HOME/pyclaw.log"
    echo "  • 错误日志: $PYCLAW_HOME/pyclaw.err.log"
    echo ""
    echo "常用命令:"
    echo "  $0 status   - 查看状态"
    echo "  $0 logs     - 查看日志"
    echo "  $0 stop     - 停止服务"
}

# 启动服务
cmd_start() {
    check_installed
    log_info "启动 PyClaw 服务..."
    launchctl start "$PLIST_NAME"
    log_success "服务已启动"
}

# 停止服务
cmd_stop() {
    check_installed
    log_info "停止 PyClaw 服务..."
    launchctl stop "$PLIST_NAME"
    log_success "服务已停止"
}

# 重启服务
cmd_restart() {
    check_installed
    log_info "重启 PyClaw 服务..."
    launchctl stop "$PLIST_NAME"
    sleep 1
    launchctl start "$PLIST_NAME"
    log_success "服务已重启"
}

# 查看状态
cmd_status() {
    if [ -f "$PLIST_PATH" ]; then
        if launchctl list | grep -q "$PLIST_NAME"; then
            log_success "服务已安装并正在运行"
            echo ""
            echo "PID 信息:"
            launchctl list | grep "$PLIST_NAME"
        else
            log_warn "服务已安装但未运行"
        fi
    else
        log_warn "服务未安装"
    fi
}

# 查看日志
cmd_logs() {
    LOG_FILE="$PYCLAW_HOME/pyclaw.log"
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        log_warn "日志文件不存在: $LOG_FILE"
    fi
}

# 卸载服务
cmd_uninstall() {
    check_installed
    log_info "卸载 PyClaw 服务..."
    launchctl stop "$PLIST_NAME" 2>/dev/null || true
    launchctl unload "$PLIST_PATH"
    rm -f "$PLIST_PATH"
    log_success "服务已卸载"
}

# 主命令
case "${1:-help}" in
    install)
        cmd_install
        ;;
    start)
        cmd_start
        ;;
    stop)
        cmd_stop
        ;;
    restart)
        cmd_restart
        ;;
    status)
        cmd_status
        ;;
    logs)
        cmd_logs
        ;;
    uninstall)
        cmd_uninstall
        ;;
    *)
        echo "PyClaw 服务管理工具"
        echo ""
        echo "用法: $0 {install|start|stop|restart|status|logs|uninstall}"
        echo ""
        echo "命令:"
        echo "  install   - 安装为 launchd 服务"
        echo "  start     - 启动服务"
        echo "  stop      - 停止服务"
        echo "  restart   - 重启服务"
        echo "  status    - 查看服务状态"
        echo "  logs      - 实时查看日志"
        echo "  uninstall - 卸载服务"
        ;;
esac
