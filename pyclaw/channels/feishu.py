from __future__ import annotations

import asyncio
import json
import uuid
from typing import Optional

import aiohttp
import lark_oapi as lark

from pyclaw.core.message import Message, MessageRole, MessageType

from .base import BaseChannel


class FeishuChannel(BaseChannel):
    """飞书消息通道 - 使用官方 SDK 长连接模式"""

    name = "feishu"

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        allowed_user_ids: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self.app_id = app_id
        self.app_secret = app_secret
        self.allowed_user_ids = allowed_user_ids
        self._session = None
        self._loop = None

    async def start(self) -> None:
        """启动飞书通道"""
        print("🤖 飞书 Bot 启动中...")

        self._loop = asyncio.get_event_loop()

        # 1. 创建 aiohttp session 用于发消息
        self._session = aiohttp.ClientSession()

        # 2. 消息处理器
        def on_message_receive(event):
            try:
                # event.event 里面有 message 和 sender
                event_obj = event.event
                message = event_obj.message
                sender = event_obj.sender

                # 消息内容
                content = json.loads(message.content)
                text = content.get("text", "")

                # 发送者信息
                open_id = sender.sender_id.open_id
                sender_type = sender.sender_type

                # 不处理机器人自己发的消息
                if sender_type == "bot":
                    return

                # 权限检查
                if self.allowed_user_ids and open_id not in self.allowed_user_ids:
                    print(f"🚫 用户 {open_id} 不在白名单中")
                    return

                # 给用户消息添加 OK 反应标签 - 表示已收到并处理
                message_id = message.message_id
                self._add_ok_reaction_sync(message_id)

                # 创建消息
                msg = Message(
                    id=str(uuid.uuid4()),
                    channel="feishu",
                    channel_user_id=open_id,
                    user_id=open_id,
                    session_id=f"feishu:{open_id}",
                    type=MessageType.TEXT,
                    role=MessageRole.USER,
                    content=text,
                )

                # 处理消息
                asyncio.run_coroutine_threadsafe(
                    self._handle_message(msg),
                    self._loop,
                )

            except Exception as e:
                print(f"❌ 处理消息失败: {e}")
                import traceback
                traceback.print_exc()

        # 3. 注册事件处理器
        # 空处理器 - 用于消掉不需要的事件错误
        def noop_handler(event):
            pass

        handler = (
            lark.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(on_message_receive)
            .register_p2_im_message_message_read_v1(noop_handler)
            .register_p2_im_chat_access_event_bot_p2p_chat_entered_v1(noop_handler)
            .build()
        )

        # 4. 启动长连接
        def start_ws():
            try:
                print("🔗 正在建立飞书长连接...")
                ws_client = lark.ws.Client(
                    self.app_id,
                    self.app_secret,
                    event_handler=handler,
                )
                ws_client.start()
            except Exception as e:
                print(f"❌ WebSocket 启动失败: {e}")

        import threading
        ws_thread = threading.Thread(target=start_ws, daemon=True)
        ws_thread.start()

        await asyncio.sleep(2)
        print("✅ 飞书长连接已启动！")

    async def stop(self) -> None:
        if self._session:
            await self._session.close()

    async def send_message(self, message: Message) -> None:
        """发送消息到飞书"""
        try:
            print(f"📤 [飞书] 正在发送消息...")

            token = await self._get_tenant_token()

            url = "https://open.feishu.cn/open-apis/im/v1/messages"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            }

            receive_id = message.channel_user_id
            content = json.dumps({"text": message.content}, ensure_ascii=False)

            params = {"receive_id_type": "open_id"}
            data = {
                "receive_id": receive_id,
                "content": content,
                "msg_type": "text",
            }

            async with self._session.post(url, headers=headers, params=params, json=data) as resp:
                result = await resp.json()

                if result.get("code") != 0:
                    print(f"❌ [飞书] 发送失败: {result.get('msg')} (code: {result.get('code')})")
                else:
                    print(f"✅ [飞书] 消息发送成功")

        except Exception as e:
            print(f"❌ [飞书] 发送异常: {e}")
            import traceback
            traceback.print_exc()

    def _add_ok_reaction_sync(self, message_id: str) -> None:
        """给用户消息添加 OK 反应标签 - 表示已收到"""
        try:
            import requests

            # 获取 tenant_token
            token_url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
            token_data = {
                "app_id": self.app_id,
                "app_secret": self.app_secret,
            }
            token_resp = requests.post(token_url, json=token_data)
            token = token_resp.json()["tenant_access_token"]

            # 添加消息反应 API
            url = f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/reactions"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            }

            # 用正确的参数格式 - 参考返回示例
            # reaction_type 是一个嵌套对象
            data = {
                "reaction_type": {
                    "emoji_type": "OK",
                }
            }

            print(f"👍 正在添加 OK 反应: message_id={message_id}")

            resp = requests.post(url, headers=headers, json=data)
            result = resp.json()

            print(f"   API 响应: code={result.get('code')}, msg={result.get('msg')}")

            if result.get("code") != 0:
                print(f"⚠️ 添加反应失败: {result}")
            else:
                print(f"✅ OK 反应已添加")

        except Exception as e:
            print(f"⚠️ 添加反应异常: {e}")

    async def _get_tenant_token(self) -> str:
        """获取 tenant_access_token"""
        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        data = {
            "app_id": self.app_id,
            "app_secret": self.app_secret,
        }

        async with self._session.post(url, json=data) as resp:
            result = await resp.json()
            if result.get("code") == 0:
                return result["tenant_access_token"]
            else:
                raise Exception(f"获取 Token 失败: {result}")
