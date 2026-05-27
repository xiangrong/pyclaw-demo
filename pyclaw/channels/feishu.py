from __future__ import annotations

import asyncio
import json
import re
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
                content_raw = message.content
                content = json.loads(content_raw)
                
                msg_type = message.message_type # text, image, file, audio, media, sticker, interactive
                text = ""
                file_path = None
                m_type = MessageType.TEXT

                if msg_type == "text":
                    text = content.get("text", "")
                elif msg_type == "post":
                    # 富文本，提取文本部分
                    content_list = content.get("content", [])
                    text_parts = []
                    for row in content_list:
                        for item in row:
                            if item.get("tag") == "text":
                                text_parts.append(item.get("text", ""))
                    text = "".join(text_parts)
                elif msg_type in ["image", "file", "audio", "media"]:
                    key = content.get("image_key") or content.get("file_key")
                    file_name = content.get("file_name", f"feishu_{msg_type}_{message.message_id}")
                    if key:
                        # 启动异步下载
                        file_path = asyncio.run_coroutine_threadsafe(
                            self._download_media(message.message_id, key, file_name, msg_type),
                            self._loop
                        ).result()
                        
                        text = f"[Received {msg_type}: {file_name}] Local path: {file_path}"
                        if msg_type == "image": m_type = MessageType.IMAGE
                        elif msg_type == "file": m_type = MessageType.FILE
                        # 允许用户附带文字描述
                        if "text" in content:
                            text += f"\nDescription: {content['text']}"
                else:
                    text = f"[Unsupported message type: {msg_type}]"

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
                    type=m_type,
                    role=MessageRole.USER,
                    content=text,
                    metadata={"file_path": file_path} if file_path else {},
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
            .register_p2_im_message_reaction_created_v1(noop_handler)
            .register_p2_im_message_reaction_deleted_v1(noop_handler)
            .build()
        )

        # 4. 启动长连接
        def start_ws():
            try:
                print("🔗 正在建立飞书长连接...")
                # 关键：在子线程中实例化 Client，确保其内部的 asyncio.Lock 等对象绑定到子线程的 loop
                ws_client = lark.ws.Client(
                    self.app_id,
                    self.app_secret,
                    event_handler=handler,
                    log_level=lark.LogLevel.WARNING, # 减少日志输出
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

    async def _upload_image(self, image_url: str) -> Optional[str]:
        """下载远程图片并上传到飞书获取 image_key"""
        try:
            token = await self._get_tenant_token()
            upload_url = "https://open.feishu.cn/open-apis/im/v1/images"
            
            # 1. 下载图片内容
            async with self._session.get(image_url) as resp:
                if resp.status != 200:
                    print(f"⚠️ 下载图片失败: {resp.status} - {image_url}")
                    return None
                image_data = await resp.read()
            
            # 2. 上传到飞书
            from aiohttp import FormData
            form_data = FormData()
            form_data.add_field('image_type', 'message')
            form_data.add_field('image', image_data, filename='chart.png')
            
            headers = {
                "Authorization": f"Bearer {token}",
            }
            
            async with self._session.post(upload_url, headers=headers, data=form_data) as resp:
                result = await resp.json()
                if result.get("code") == 0:
                    return result["data"]["image_key"]
                else:
                    print(f"⚠️ 上传图片到飞书失败: {result}")
        except Exception as e:
            print(f"⚠️ 上传图片过程出现异常: {e}")
        return None

    async def send_message(self, message: Message) -> None:
        """发送消息到飞书 (支持文本和消息卡片，自动处理图片)"""
        try:
            print(f"📤 [飞书] 正在发送消息...")

            token = await self._get_tenant_token()
            url = "https://open.feishu.cn/open-apis/im/v1/messages"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            }

            receive_id = message.channel_user_id
            content_str = message.content
            
            # 检测是否包含 Markdown 格式，如果是则发送消息卡片以获得更好的显示效果
            if self._is_markdown(content_str):
                msg_type = "interactive"
                
                # 尝试提取标题
                header = None
                header_match = re.search(r"^#\s+(.*)$", content_str, re.MULTILINE)
                if header_match:
                    header = header_match.group(1)
                    # 从正文中移除该标题，避免重复显示
                    content_str = content_str.replace(header_match.group(0), "").strip()

                # 匹配 ![alt](url)
                img_pattern = r"!\[(.*?)\]\((https?://.*?)\)"
                
                elements = []
                last_idx = 0
                
                # 查找所有图片并交替构建文本和图片元素
                matches = list(re.finditer(img_pattern, content_str))
                
                if not matches:
                    elements.append({"tag": "markdown", "content": content_str})
                else:
                    for match in matches:
                        # 1. 添加图片之前的文本
                        text_before = content_str[last_idx:match.start()].strip()
                        if text_before:
                            elements.append({"tag": "markdown", "content": text_before})
                        
                        alt_text = match.group(1)
                        img_url = match.group(2)
                        
                        # 2. 尝试上传图片到飞书
                        image_key = await self._upload_image(img_url)
                        if image_key:
                            elements.append({
                                "tag": "img",
                                "img_key": image_key,
                                "alt": {"tag": "plain_text", "content": alt_text or "image"},
                                "mode": "fit_horizontal",
                                "preview": True
                            })
                        else:
                            elements.append({"tag": "markdown", "content": f"![{alt_text}]({img_url})"})
                        
                        last_idx = match.end()
                    
                    text_remaining = content_str[last_idx:].strip()
                    if text_remaining:
                        elements.append({"tag": "markdown", "content": text_remaining})

                # 构建交互式卡片
                card_content = {
                    "config": {"wide_screen_mode": True},
                    "elements": elements
                }
                
                if header:
                    card_content["header"] = {
                        "template": "blue",
                        "title": {"tag": "plain_text", "content": header}
                    }
                
                # 添加页脚
                card_content["elements"].append({
                    "tag": "note",
                    "elements": [{"tag": "plain_text", "content": "🤖 Powered by PyClaw"}]
                })

                content = json.dumps(card_content, ensure_ascii=False)
            else:
                msg_type = "text"
                content = json.dumps({"text": content_str}, ensure_ascii=False)

            params = {"receive_id_type": "open_id"}
            data = {
                "receive_id": receive_id,
                "content": content,
                "msg_type": msg_type,
            }

            async with self._session.post(url, headers=headers, params=params, json=data) as resp:
                result = await resp.json()

                if result.get("code") != 0:
                    print(f"❌ [飞书] 发送失败: {result.get('msg')} (code: {result.get('code')})")
                    # 如果卡片发送失败（可能因为内容过长或格式问题），尝试退回到普通文本
                    if msg_type == "interactive":
                        print("⚠️ 卡片发送失败，尝试退回到纯文本发送...")
                        data["msg_type"] = "text"
                        data["content"] = json.dumps({"text": message.content}, ensure_ascii=False)
                        async with self._session.post(url, headers=headers, params=params, json=data) as retry_resp:
                            retry_result = await retry_resp.json()
                            if retry_result.get("code") == 0:
                                print("✅ [飞书] 纯文本重试发送成功")
                else:
                    print(f"✅ [飞书] 消息发送成功 ({msg_type})")

        except Exception as e:
            print(f"❌ [飞书] 发送异常: {e}")
            import traceback
            traceback.print_exc()

    def _is_markdown(self, text: str) -> bool:
        """简单检测文本是否包含 Markdown 格式"""
        markers = ["**", "###", "##", "- ", "1. ", "[", "`"]
        return any(marker in text for marker in markers)

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

    async def _download_media(self, message_id: str, file_key: str, file_name: str, msg_type: str) -> str:
        """从飞书下载媒体文件并保存到本地"""
        import os
        from pathlib import Path

        # 确保下载目录存在
        download_dir = Path.home() / ".pyclaw" / "downloads" / "feishu"
        download_dir.mkdir(parents=True, exist_ok=True)
        
        local_path = download_dir / f"{message_id}_{file_name}"
        
        try:
            print(f"📥 [飞书] 正在下载媒体文件: {file_name} ({msg_type})...")
            
            token = await self._get_tenant_token()
            
            # 使用官方接口下载媒体文件
            # 对于 image 使用 get_image，对于其他使用 get_file
            if msg_type == "image":
                url = f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/resources/{file_key}?type=image"
            else:
                url = f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/resources/{file_key}?type=file"

            headers = {
                "Authorization": f"Bearer {token}",
            }

            async with self._session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    with open(local_path, "wb") as f:
                        f.write(await resp.read())
                    print(f"✅ [飞书] 文件已下载到: {local_path}")
                    return str(local_path)
                else:
                    error_text = await resp.text()
                    print(f"❌ [飞书] 下载失败 (HTTP {resp.status}): {error_text}")
        except Exception as e:
            print(f"❌ [飞书] 下载异常: {e}")
            import traceback
            traceback.print_exc()
            
        return ""
