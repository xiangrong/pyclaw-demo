from __future__ import annotations

import uuid

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    filters,
)

from pyclaw.core.message import Message, MessageRole, MessageType

from .base import BaseChannel


class TelegramChannel(BaseChannel):
    """Telegram 消息通道"""

    name = "telegram"

    def __init__(self, token: str, allowed_user_ids: list[int] | None = None) -> None:
        super().__init__()
        self.token = token
        self.allowed_user_ids = allowed_user_ids  # 如果为None，则允许所有用户
        self._app = None

    async def start(self) -> None:
        """启动Telegram Bot"""
        self._app = ApplicationBuilder().token(self.token).build()

        # 添加消息处理器
        handler = MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text_message)
        self._app.add_handler(handler)

        # 添加命令处理器
        self._app.add_handler(MessageHandler(filters.COMMAND, self._on_command))

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()

        print("🤖 Telegram Bot started!")

    async def stop(self) -> None:
        """停止Telegram Bot"""
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()

    async def _on_text_message(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """处理文本消息"""
        if not update.message or not update.effective_user:
            return

        # 检查用户权限
        user_id = update.effective_user.id
        if self.allowed_user_ids and user_id not in self.allowed_user_ids:
            await update.message.reply_text("❌ 你没有权限使用此Bot")
            return

        msg = Message(
            id=str(uuid.uuid4()),
            channel="telegram",
            channel_user_id=str(user_id),
            user_id=str(user_id),
            session_id=f"telegram:{user_id}",
            type=MessageType.TEXT,
            role=MessageRole.USER,
            content=update.message.text or "",
        )

        await self._handle_message(msg)

    async def _on_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """处理命令"""
        if not update.message or not update.effective_user:
            return

        text = update.message.text or ""
        if text == "/start":
            await update.message.reply_text(
                "👋 欢迎使用 PyClaw AI Agent！\n"
                "有什么我可以帮你的吗？\n\n"
                "我可以帮你执行命令、读写文件等任务。",
            )
        elif text == "/help":
            await update.message.reply_text(
            "📖 可用命令：\n"
            "/start - 开始使用\n"
            "/help - 显示帮助\n"
            "/clear - 清空会话历史",
            )

    async def send_message(self, message: Message) -> None:
        """发送消息到Telegram"""
        if not self._app:
            return

        chat_id = int(message.channel_user_id)
        text = message.content

        # 长消息分块发送
        chunks = [text[i:i + 4000] for i in range(0, len(text), 4000)]
        for chunk in chunks:
            await self._app.bot.send_message(
                chat_id=chat_id,
                text=chunk,
                parse_mode=None,
            )
