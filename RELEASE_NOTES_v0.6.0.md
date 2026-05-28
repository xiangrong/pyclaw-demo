# Release Notes - v0.6.0

## 🚀 PyClaw v0.6.0: 官方个人微信通道正式上线！

本次更新带来了里程碑式的进展：PyClaw 正式支持通过腾讯官方 **iLink Bot (ClawBot)** 协议接入个人号微信，无需逆向，合规稳定，且支持扫码快速接入。

### 🌟 新增功能

- **📱 官方微信个人号支持**：新增 `WechatChannel`。基于官方 iLink 协议，支持私聊对话，极大降低封号风险。
- **🔗 自动会话管理**：内置 `context_token` 自动追踪机制。iLink 协议要求严格的“回复型”对话，PyClaw 自动管理这些复杂的 Session Token，确保消息回复 100% 成功。
- **🎯 自动捕获 Bot ID**：登录流程大幅优化。即使 API 响应不完整，系统也会从第一条收到的消息中自动提取真实的 `bot_id` 并提示用户保存。
- **🛡️ 稳定长连接**：采用 Long Polling 机制。无需公网 IP，无需 ngrok，内网服务器即可轻松接入。

### 🛠️ 改进与优化

- **📦 依赖更新**：新增 `qrcode` (终端显示) 和 `cryptography` (媒体加密准备) 依赖。
- **📝 文档升级**：更新了 `README.md` 和 `GEMINI.md`，提供了详尽的微信配置向导。
- **⚙️ CLI 增强**：`pyclaw init` 模板已包含微信配置项。

### 📝 接入三步走

1.  **升级**：执行 `pip install .`。
2.  **启动**：在 `config.yaml` 中开启 `wechat` 配置，运行 `pyclaw start` 扫码。
3.  **持久化**：扫码后给机器人发消息，复制终端显示的 `bot_token` 和 `bot_id` 到配置文件，即可实现免扫码启动。

---

**Full Changelog**: https://github.com/xiangrong/pyclaw-demo/compare/v0.5.0...v0.6.0
