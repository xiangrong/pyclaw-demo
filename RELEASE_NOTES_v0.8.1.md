# Release Notes - v0.8.1

## 🐛 Bug Fixes & Polishing

This is a maintenance release following v0.8.0, focusing on critical fixes for cross-channel file delivery and improving operator visibility.

### 🌟 Key Changes

- **📁 WeChat (iLink) File Delivery**: Fully implemented the official WeChat iLink CDN upload protocol. 
    - Supports local **AES-128-ECB encryption** before upload.
    - Integrated with Tencent CDN for reliable large file transfer.
    - Supports `upload_full_url` pre-signed URL parsing.
    - Added graceful text-based fallback for text files if CDN upload fails.
- **📁 Feishu (Lark) Fixes**: Resolved a missing `import os` issue that caused file uploads to crash.
- **🔄 Agent Loop Visibility**: Restored the `🛠️ [Tool Call]` log entries in the agent loop. You can once again see exactly which tools and arguments the agent is choosing in real-time.
- **🧹 Clean Logs**: Removed unnecessary debug payload logging in the WeChat channel.

### 📝 Release Info
- **Version**: `0.8.1`
- **Build Artifacts**:
  - `dist/pyclaw-0.8.1-py3-none-any.whl`
  - `dist/pyclaw-0.8.1.tar.gz`

---
**PyClaw: Connect, Reason, Act.**
