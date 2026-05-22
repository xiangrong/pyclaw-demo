from __future__ import annotations

import os
import uuid
from typing import Optional

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

    def __init__(self, token: str, allowed_user_ids: Optional[list[int]] = None) -> None:
        super().__init__()
        self.token = token
        self.allowed_user_ids = allowed_user_ids  # 如果为None，则允许所有用户
        self._app = None

    async def start(self) -> None:
        """启动Telegram Bot"""
        self._app = ApplicationBuilder().token(self.token).build()

        # 添加消息处理器 (支持文本和文件)
        handler = MessageHandler((filters.TEXT | filters.Document.ALL) & ~filters.COMMAND, self._on_message)
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

    async def _on_message(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """处理文本和文件消息"""
        if not update.message or not update.effective_user:
            return

        # 检查用户权限
        user_id = update.effective_user.id
        if self.allowed_user_ids and user_id not in self.allowed_user_ids:
            await update.message.reply_text("❌ 你没有权限使用此Bot")
            return

        content = ""
        msg_type = MessageType.TEXT

        # 处理文件上传
        if update.message.document:
            document = update.message.document
            file_name = document.file_name or "unknown_file"
            
            # 创建下载目录
            download_dir = os.path.abspath(os.path.join(os.getcwd(), "downloads"))
            os.makedirs(download_dir, exist_ok=True)
            
            file_path = os.path.join(download_dir, file_name)
            
            # 下载文件
            telegram_file = await context.bot.get_file(document.file_id)
            await telegram_file.download_to_drive(file_path)
            
            # 构造提示词，告诉 LLM 文件存在哪里
            caption = update.message.caption or ""
            content = f"[User uploaded a file named '{file_name}'. It has been saved locally at: {file_path}]\n"
            if caption:
                content += f"User message: {caption}"
            
            msg_type = MessageType.FILE
        elif update.message.text:
            content = update.message.text
        
        if not content:
            return

        msg = Message(
            id=str(uuid.uuid4()),
            channel="telegram",
            channel_user_id=str(user_id),
            user_id=str(user_id),
            session_id=f"telegram:{user_id}",
            type=msg_type,
            role=MessageRole.USER,
            content=content,
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

    async def send_stream(
        self,
        stream,
        channel_user_id: str,
    ) -> str:
        """流式发送消息"""
        if not self._app:
            return ""

        chat_id = int(channel_user_id)
        full_content = ""
        last_update_time = 0
        update_interval = 0.5  # 每0.5秒更新一次消息

        # 发送第一条"思考中..."消息
        message = await self._app.bot.send_message(
            chat_id=chat_id,
            text="🤔 思考中...",
        )
        message_id = message.message_id

        try:
            async for chunk in stream:
                full_content += chunk

                # 节流：避免太频繁的 API 调用
                current_time = asyncio.get_event_loop().time()
                if current_time - last_update_time > update_interval:
                    if full_content.strip():  # 确保有内容才更新
                        await self._app.bot.edit_message_text(
                            chat_id=chat_id,
                            message_id=message_id,
                            text=full_content,
                            parse_mode=None,
                        )
                        last_update_time = current_time

            # 流式结束，确保最后更新一次
            if full_content.strip():
                await self._app.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=full_content,
                    parse_mode=None,
                )

        except Exception as e:
            print(f"⚠️ Stream error: {e}")
            # 出错时发送完整内容
            if full_content.strip():
                await self._app.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=full_content,
                    parse_mode=None,
                )

        return full_content
