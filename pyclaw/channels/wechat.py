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
        
        # 自动捕获 bot_id (如果当前为空)
        if not self.bot_id and to_user_id:
            self.bot_id = to_user_id
            print(f"🎯 自动捕获到 Bot ID: {self.bot_id}")
            print(f"请将其更新到 config.yaml: bot_id: \"{self.bot_id}\"")

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
            id=data.get("msg_id", str(uuid.uuid4())),
            channel="wechat",
            channel_user_id=from_user_id,
            user_id=from_user_id,
            session_id=f"wechat:{from_user_id}",
            type=msg_type,
            role=MessageRole.USER,
            content=content,
            metadata={"context_token": context_token}
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
