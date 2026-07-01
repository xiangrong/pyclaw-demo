import pytest

from pyclaw.channels.base import BaseChannel
from pyclaw.channels.wechat import WechatChannel


class DummyChannel(BaseChannel):
    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def send_message(self, message) -> None:
        pass

    async def send_file(self, channel_user_id, file_path, description=None, metadata=None) -> None:
        pass


def test_source_message_id_dedupe_returns_false_for_duplicate():
    channel = DummyChannel()

    assert channel._remember_source_message_id("msg-1") is True
    assert channel._remember_source_message_id("msg-1") is False
    assert channel._remember_source_message_id("msg-2") is True


def test_message_fingerprint_dedupe_catches_same_content_with_different_ids():
    channel = DummyChannel()

    assert channel._remember_message_fingerprint("user-1", "text", " 世界杯最新赛事 ") is True
    assert channel._remember_message_fingerprint("user-1", "text", "世界杯最新赛事") is False
    assert channel._remember_message_fingerprint("user-1", "text", "世界杯最新赛程") is True


def test_message_fingerprint_dedupe_is_scoped_by_sender():
    channel = DummyChannel()

    assert channel._remember_message_fingerprint("user-1", "text", "hello") is True
    assert channel._remember_message_fingerprint("user-2", "text", "hello") is True


@pytest.mark.asyncio
async def test_wechat_duplicate_msg_id_is_ignored():
    channel = WechatChannel(bot_token="token", bot_id="bot")
    handled = []

    async def handler(message):
        handled.append(message)

    channel.on_message(handler)
    data = {
        "msg_id": "wechat-msg-1",
        "from_user_id": "user-1",
        "to_user_id": "bot",
        "context_token": "ctx",
        "item_list": [{"type": 1, "text_item": {"text": "hello"}}],
    }

    await channel._process_incoming_message(data)
    await channel._process_incoming_message(data)

    assert len(handled) == 1
    assert handled[0].id == "wechat-msg-1"
    assert handled[0].metadata["source_message_id"] == "wechat-msg-1"


@pytest.mark.asyncio
async def test_wechat_duplicate_content_with_different_msg_id_is_ignored():
    channel = WechatChannel(bot_token="token", bot_id="bot")
    handled = []

    async def handler(message):
        handled.append(message)

    channel.on_message(handler)

    base_data = {
        "from_user_id": "user-1",
        "to_user_id": "bot",
        "context_token": "ctx",
        "item_list": [{"type": 1, "text_item": {"text": "帮我修改代码"}}],
    }
    await channel._process_incoming_message({**base_data, "msg_id": "wechat-msg-1"})
    await channel._process_incoming_message({**base_data, "msg_id": "wechat-msg-2"})

    assert len(handled) == 1
    assert handled[0].id == "wechat-msg-1"


@pytest.mark.asyncio
async def test_wechat_duplicate_content_without_msg_id_is_ignored():
    channel = WechatChannel(bot_token="token", bot_id="bot")
    handled = []

    async def handler(message):
        handled.append(message)

    channel.on_message(handler)
    data = {
        "from_user_id": "user-1",
        "to_user_id": "bot",
        "context_token": "ctx",
        "item_list": [{"type": 1, "text_item": {"text": "继续"}}],
    }

    await channel._process_incoming_message(data)
    await channel._process_incoming_message(data)

    assert len(handled) == 1


@pytest.mark.asyncio
async def test_wechat_self_message_is_ignored():
    channel = WechatChannel(bot_token="token", bot_id="bot")
    handled = []

    async def handler(message):
        handled.append(message)

    channel.on_message(handler)

    await channel._process_incoming_message({
        "msg_id": "wechat-msg-from-bot",
        "from_user_id": "bot",
        "to_user_id": "user-1",
        "item_list": [{"type": 1, "text_item": {"text": "loop"}}],
    })

    assert handled == []
