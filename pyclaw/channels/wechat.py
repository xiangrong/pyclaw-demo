from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import time
import uuid
from typing import Any, AsyncGenerator, Optional

import aiohttp
import qrcode
from pyclaw.core.message import Message, MessageRole, MessageType

from .base import BaseChannel


class WechatChannel(BaseChannel):
    """微信个人号消息通道 (iLink Bot API / ClawBot)"""

    name = "wechat"
    BASE_URL = "https://ilinkai.weixin.qq.com"

    def __init__(
        self,
        bot_token: Optional[str] = None,
        bot_id: Optional[str] = None,
        allowed_user_ids: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self.bot_token = bot_token
        self.bot_id = bot_id
        self.allowed_user_ids = allowed_user_ids
        self.session: Optional[aiohttp.ClientSession] = None
        self.get_updates_buf: str = ""
        self._running = False
        # 生成一个固定的 X-WECHAT-UIN 供本次会话使用
        random_uint32 = random.randint(100000000, 999999999)
        self.x_wechat_uin = base64.b64encode(str(random_uint32).encode()).decode()
        # 存储每个用户的最新 context_token
        self.context_tokens: dict[str, str] = {}

    def _get_headers(self) -> dict[str, str]:
        """生成请求头"""
        headers = {
            "Content-Type": "application/json",
            "X-WECHAT-UIN": self.x_wechat_uin,
        }
        if self.bot_token:
            headers["Authorization"] = f"Bearer {self.bot_token}"
            headers["AuthorizationType"] = "ilink_bot_token"
        
        return headers

    async def _login(self) -> None:
        """扫码登录流程"""
        if not self.session:
            return

        print("📢 正在获取微信登录二维码...")
        async with self.session.get(
            f"{self.BASE_URL}/ilink/bot/get_bot_qrcode?bot_type=3",
            headers=self._get_headers()
        ) as resp:
            data = await resp.json(content_type=None)
            # iLink API 可能使用 ret 作为错误码，0 表示成功
            ret = data.get("ret", data.get("errcode", -1))
            if ret != 0:
                print(f"❌ 获取二维码失败: {data}")
                return
            
            qrcode_url = data.get("qrcode_url") or data.get("qrcode_img_content")
            qrcode_id = data.get("qrcode")
            
            if not qrcode_url or not qrcode_id:
                print(f"❌ 响应中缺少关键信息: {data}")
                return
            
            # 显示二维码
            qr = qrcode.QRCode()
            qr.add_data(qrcode_url)
            print("\n请使用微信扫码登录：")
            qr.print_ascii()
            print(f"或者打开此链接扫码: {qrcode_url}\n")

        # 轮询扫码状态
        while True:
            async with self.session.get(
                f"{self.BASE_URL}/ilink/bot/get_qrcode_status?qrcode={qrcode_id}",
                headers=self._get_headers()
            ) as resp:
                data = await resp.json(content_type=None)
                status = data.get("status")
                ret = data.get("ret", data.get("errcode", 0))
                
                # 添加调试输出，帮助排查扫码后的状态
                # print(f"DEBUG: QR Status Check - status={status}, ret={ret}, data={data}")
                
                if status == 2:  # 已确认
                    self.bot_token = data.get("bot_token")
                    self.bot_id = data.get("ilink_bot_id") or data.get("bot_id")
                    print(f"✅ 登录成功！完整响应: {data}")
                    if self.bot_token:
                        print(f"\n💡 建议将以下配置保存到 config.yaml 以免重复扫码：")
                        print(f"wechat:")
                        print(f"  bot_token: \"{self.bot_token}\"")
                        print(f"  bot_id: \"{self.bot_id or ''}\"")
                        break
                    else:
                        print(f"⚠️ 扫码确认成功但未获取到 Token: {data}")
                elif status == 3:  # 已过期
                    print("❌ 二维码已过期，请重新运行。")
                    return
                elif status == 1:  # 已扫码未确认
                    if not hasattr(self, "_last_status") or self._last_status != 1:
                        print("📱 已扫码，请在手机上点击确认...")
                    self._last_status = 1
                elif status == 0: # 等待扫码
                    pass
                else:
                    # 某些 API 可能会在扫码确认后返回不同的 ret 或 status
                    if ret == 0 and data.get("bot_token"):
                        self.bot_token = data["bot_token"]
                        self.bot_id = data.get("ilink_bot_id") or data.get("bot_id")
                        print(f"✅ 登录成功 (直接获取)！完整响应: {data}")
                        print(f"\n💡 建议将以下配置保存到 config.yaml 以免重复扫码：")
                        print(f"wechat:")
                        print(f"  bot_token: \"{self.bot_token}\"")
                        print(f"  bot_id: \"{self.bot_id or ''}\"")
                        break
                
                await asyncio.sleep(2)

    async def start(self) -> None:
        """启动微信通道"""
        self.session = aiohttp.ClientSession()
        self._running = True

        if not self.bot_token:
            await self._login()

        if not self.bot_token:
            print("❌ 微信登录失败，通道无法启动。")
            return

        print("🤖 Wechat Channel started (iLink Bot)!")
        
        # 启动接收循环
        asyncio.create_task(self._poll_messages())

    async def stop(self) -> None:
        """停止微信通道"""
        self._running = False
        if self.session:
            await self.session.close()

    async def _poll_messages(self) -> None:
        """消息拉取循环"""
        while self._running:
            try:
                payload = {
                    "get_updates_buf": self.get_updates_buf
                }
                async with self.session.post(
                    f"{self.BASE_URL}/ilink/bot/getupdates",
                    headers=self._get_headers(),
                    json=payload,
                    timeout=40  # 接口本身通常 35s 超时
                ) as resp:
                    if resp.status != 200:
                        await asyncio.sleep(5)
                        continue
                        
                    data = await resp.json(content_type=None)
                    errcode = data.get("errcode", data.get("ret", 0))
                    if errcode != 0:
                        if errcode == -14:  # session timeout
                            print("⚠️ 微信会话已过期，正在尝试重新登录...")
                            self.bot_token = None
                            self.get_updates_buf = ""
                            await self._login()
                            continue
                        
                        print(f"⚠️ GetUpdates error: {data}")
                        await asyncio.sleep(5)
                        continue
                    
                    # 更新游标 (iLink API 使用 sync_buf)
                    self.get_updates_buf = data.get("sync_buf", data.get("get_updates_buf", ""))
                    
                    # 处理消息
                    messages = data.get("msgs", [])
                    for msg_data in messages:
                        await self._process_incoming_message(msg_data)
                        
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"❌ Error in WeChat poll: {e}")
                await asyncio.sleep(5)

    async def _process_incoming_message(self, data: dict[str, Any]) -> None:
        """处理收到的原始消息"""
        from_user_id = data.get("from_user_id", "")
        to_user_id = data.get("to_user_id", "")
        context_token = data.get("context_token", "")
        msg_id = data.get("msg_id", "")
        
        # 自动捕获 bot_id (如果当前为空)
        if not self.bot_id and to_user_id:
            self.bot_id = to_user_id
            print(f"🎯 自动捕获到 Bot ID: {self.bot_id}")
            print(f"请将其更新到 config.yaml: bot_id: \"{self.bot_id}\"")

        # 不处理机器人自己发出的消息，避免平台回投导致自触发循环。
        if self.bot_id and from_user_id == self.bot_id:
            return

        # iLink polling/reconnect may redeliver a message with the same msg_id.
        # Only dedupe when the upstream provides a stable id so users can still
        # intentionally send the same text twice on platforms without msg_id.
        if msg_id and not self._remember_source_message_id(msg_id):
            print(f"↩️ [WeChat] 忽略重复消息: {msg_id}")
            return

        if context_token:
            self.context_tokens[from_user_id] = context_token

        # 权限检查
        if self.allowed_user_ids and from_user_id not in self.allowed_user_ids:
            return

        content = ""
        msg_type = MessageType.TEXT
        
        # 处理消息项列表 (iLink API 使用 item_list)
        items = data.get("item_list", [])
        for item in items:
            if item.get("type") == 1:  # 文本类型
                text_item = item.get("text_item", {})
                content += text_item.get("text", "")
            elif item.get("type") == 34:  # 语音
                voice_item = item.get("voice_item", {})
                content += voice_item.get("text", "[语音消息]")
        
        if not content:
            return

        msg = Message(
            id=msg_id or str(uuid.uuid4()),
            channel="wechat",
            channel_user_id=from_user_id,
            user_id=from_user_id,
            session_id=f"wechat:{from_user_id}",
            type=msg_type,
            role=MessageRole.USER,
            content=content,
            metadata={"context_token": context_token, "source_message_id": msg_id}
        )

        await self._handle_message(msg)

    async def send_message(self, message: Message) -> None:
        """发送消息到微信 (基于 iLink 官方回复规范)"""
        if not self.session or not self.bot_token:
            return

        # 必须带上对应用户的 context_token (iLink 是严格的“回复型”协议)
        context_token = message.metadata.get("context_token") or self.context_tokens.get(message.channel_user_id)
        
        if not context_token:
            print(f"⚠️ 无法发送消息给 {message.channel_user_id}: 缺少 context_token (iLink 要求必须基于用户的最后一条消息回复)")
            return

        payload = {
            "msg": {
                "from_user_id": self.bot_id or "",
                "to_user_id": message.channel_user_id,
                "client_id": str(uuid.uuid4()),
                "message_type": 2,  # 2 表示 BOT
                "message_state": 2, # 2 表示 FINISH
                "context_token": context_token,
                "item_list": [
                    {
                        "type": 1,
                        "text_item": {
                            "text": message.content
                        }
                    }
                ]
            },
            "base_info": {
                "channel_version": "2.4.1"
            }
        }

        async with self.session.post(
            f"{self.BASE_URL}/ilink/bot/sendmessage",
            headers=self._get_headers(),
            json=payload
        ) as resp:
            data = await resp.json(content_type=None)
            ret = data.get("ret", data.get("errcode", 0))
            if ret != 0:
                print(f"❌ 微信消息发送失败: {data}")

    async def send_file(
        self,
        channel_user_id: str,
        file_path: str,
        description: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """发送本地文件到微信 (由于 iLink Bot API 暂无公开的标准文件上传接口，此处采取文本降级策略)"""
        if not self.session or not self.bot_token:
            return

        try:
            # 1. 获取 context_token
            context_token = (metadata or {}).get("context_token") or self.context_tokens.get(channel_user_id)
            if not context_token:
                print(f"⚠️ [WeChat] 缺少 context_token，无法发送文件")
                return

            
            print(f"📤 [WeChat] 正在处理文件: {os.path.basename(file_path)}...")
            
            cdn_info = await self._upload_file_to_cdn(channel_user_id, file_path)
            
            if cdn_info:
                # Send file message
                import time
                import random
                client_id = f"wechat-ilink:{int(time.time()*1000)}-{os.urandom(4).hex()}"
                
                payload = {
                    "msg": {
                        "from_user_id": "",
                        "to_user_id": channel_user_id,
                        "client_id": client_id,
                        "message_type": 2,
                        "message_state": 2,
                        "context_token": context_token,
                        "item_list": [
                            {
                                "type": 4,
                                "file_item": {
                                    "media": {
                                        "encrypt_query_param": cdn_info["encrypt_query_param"],
                                        "aes_key": cdn_info["aes_key"],
                                        "encrypt_type": 1
                                    },
                                    "file_name": os.path.basename(file_path),
                                    "len": str(cdn_info["fileSize"])
                                }
                            }
                        ]
                    },
                    "base_info": {
                        "channel_version": "2.4.1"
                    }
                }
                
                # if there is a description, maybe we should send it as a text message first?
                # WeChat file item cannot have caption. We can send text first, then file.
                if description:
                    text_payload = {
                        "msg": {
                            "from_user_id": self.bot_id or "",
                            "to_user_id": channel_user_id,
                            "client_id": str(uuid.uuid4()),
                            "message_type": 2,
                            "message_state": 2,
                            "context_token": context_token,
                            "item_list": [{"type": 1, "text_item": {"text": description}}]
                        },
                        "base_info": {"channel_version": "2.4.1"}
                    }
                    await self.session.post(f"{self.BASE_URL}/ilink/bot/sendmessage", headers=self._get_headers(), json=text_payload)
                    
                async with self.session.post(
                    f"{self.BASE_URL}/ilink/bot/sendmessage",
                    headers=self._get_headers(),
                    json=payload
                ) as resp:
                    data = await resp.json(content_type=None)
                    if data.get("ret", data.get("errcode", 0)) == 0:
                        print(f"✅ [WeChat] 文件发送成功")
                        return
                    else:
                        print(f"❌ [WeChat] 微信文件发送失败: {data}")
                        
            print(f"⚠️ [WeChat] CDN 上传失败或出错，尝试使用降级策略...")


            # iLink Bot API 暂无可靠的文件上传接口 (/ilink/bot/upload_file 实际上报 404)
            # 这里采取优雅降级策略：如果是文本类型，直接发送内容；如果是二进制类型，提示不支持。
            text_extensions = {".md", ".txt", ".csv", ".json", ".py", ".js", ".html", ".css", ".yaml", ".yml", ".sh", ".log"}
            ext = os.path.splitext(file_path)[1].lower()

            if ext in text_extensions:
                print(f"ℹ️ [WeChat] 采用降级策略：以文本形式发送文件内容 ({os.path.basename(file_path)})")
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # 微信单条消息通常有限制，简单截断
                if len(content) > 2000:
                    content = content[:2000] + "\n\n...(内容过长已截断)..."

                msg_text = f"📄 【文件内容: {os.path.basename(file_path)}】\n\n{content}"
                if description:
                    msg_text = f"{description}\n\n{msg_text}"

                payload = {
                    "msg": {
                        "from_user_id": self.bot_id or "",
                        "to_user_id": channel_user_id,
                        "client_id": str(uuid.uuid4()),
                        "message_type": 2,
                        "message_state": 2,
                        "context_token": context_token,
                        "item_list": [
                            {
                                "type": 1,
                                "text_item": {
                                    "text": msg_text
                                }
                            }
                        ]
                    },
                    "base_info": {
                        "channel_version": "2.4.1"
                    }
                }
            else:
                print(f"ℹ️ [WeChat] 采用降级策略：提示用户不支持该文件类型 ({os.path.basename(file_path)})")
                msg_text = f"⚠️ 微信通道暂不支持直接接收 [{os.path.basename(file_path)}] 这种格式的文件，请通过其他通道(如飞书/Telegram)获取。"
                if description:
                    msg_text = f"{description}\n\n{msg_text}"

                payload = {
                    "msg": {
                        "from_user_id": self.bot_id or "",
                        "to_user_id": channel_user_id,
                        "client_id": str(uuid.uuid4()),
                        "message_type": 2,
                        "message_state": 2,
                        "context_token": context_token,
                        "item_list": [
                            {
                                "type": 1,
                                "text_item": {
                                    "text": msg_text
                                }
                            }
                        ]
                    },
                    "base_info": {
                        "channel_version": "2.4.1"
                    }
                }

            async with self.session.post(
                f"{self.BASE_URL}/ilink/bot/sendmessage",
                headers=self._get_headers(),
                json=payload
            ) as resp:
                data = await resp.json(content_type=None)
                ret = data.get("ret", data.get("errcode", 0))
                if ret == 0:
                    print(f"✅ [WeChat] 文件(降级文本)发送成功")
                else:
                    print(f"❌ [WeChat] 微信文件发送失败: {data}")

        except Exception as e:
            print(f"❌ [WeChat] 发送文件异常: {e}")

    async def send_stream(
        self,
        stream: AsyncGenerator[str, None],
        channel_user_id: str,
    ) -> str:
        """微信不支持原生的流式更新，此处采用分段合并发送"""
        full_content = ""
        async for chunk in stream:
            full_content += chunk
        
        # 构造一个临时 Message 对象用于发送
        msg = Message(
            id=str(uuid.uuid4()),
            channel="wechat",
            channel_user_id=channel_user_id,
            session_id="",
            type=MessageType.TEXT,
            role=MessageRole.ASSISTANT,
            content=full_content
        )
        await self.send_message(msg)
        return full_content


    async def _upload_file_to_cdn(self, channel_user_id: str, file_path: str) -> Optional[dict[str, str]]:
        """AES 加密并上传到微信 CDN"""
        try:
            import os
            import hashlib
            import base64
            import uuid
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives import padding

            with open(file_path, "rb") as f:
                plaintext = f.read()
            
            rawsize = len(plaintext)
            rawfilemd5 = hashlib.md5(plaintext).hexdigest()
            
            # AES encryption
            raw_key = os.urandom(16)
            padder = padding.PKCS7(128).padder()
            padded_data = padder.update(plaintext) + padder.finalize()
            cipher = Cipher(algorithms.AES(raw_key), modes.ECB(), backend=default_backend())
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()
            
            filesize = len(ciphertext)
            hex_key = raw_key.hex()
            filekey = rawfilemd5 + str(uuid.uuid4())[:8]
            
            # getuploadurl
            url = f"{self.BASE_URL}/ilink/bot/getuploadurl"
            payload = {
                "filekey": filekey,
                "media_type": 3,
                "to_user_id": channel_user_id,
                "rawsize": rawsize,
                "rawfilemd5": rawfilemd5,
                "filesize": filesize,
                "aeskey": hex_key,
                "no_need_thumb": True,
                "base_info": {
                    "channel_version": "2.4.1"
                }
            }
            
            async with self.session.post(url, headers=self._get_headers(), json=payload) as resp:
                data = await resp.json(content_type=None)
                if not data:
                    print(f"⚠️ [WeChat] getuploadurl 返回为空")
                    return None
                
                # Some versions return `upload_full_url`, others return `ret: 0` and `upload_param`
                if "upload_full_url" in data:
                    cdn_url = data["upload_full_url"]
                    from urllib.parse import urlparse, parse_qs
                    parsed = urlparse(cdn_url)
                    qs = parse_qs(parsed.query)
                    upload_param = qs.get("encrypted_query_param", [""])[0].replace(" ", "+")
                elif data.get("ret") == 0 and "upload_param" in data:
                    upload_param_b64 = data["upload_param"]
                    upload_param = base64.b64decode(upload_param_b64).decode('utf-8')
                    cdn_url = upload_param if upload_param.startswith("http") else f"https://novac2c.cdn.weixin.qq.com/c2c/{upload_param}"
                else:
                    print(f"⚠️ [WeChat] getuploadurl 失败: {data}")
                    return None
                
            # CDN upload
            cdn_headers = {"Content-Type": "application/octet-stream"}
            async with self.session.post(cdn_url, headers=cdn_headers, data=ciphertext) as cdn_resp:
                if cdn_resp.status != 200:
                    print(f"⚠️ [WeChat] CDN 上传失败: HTTP {cdn_resp.status}")
                    return None
                    
                encrypt_query_param = cdn_resp.headers.get("x-encrypted-param") or upload_param
                
            return {
                "encrypt_query_param": encrypt_query_param,
                "aes_key": base64.b64encode(hex_key.encode('utf-8')).decode('utf-8'),
                "fileSize": str(rawsize)
            }
        except Exception as e:
            print(f"⚠️ [WeChat] CDN 上传异常: {e}")
            return None
