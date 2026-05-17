#!/usr/bin/env python3
"""测试所有模块能否正常导入"""

import sys

print("🔍 Testing PyClaw imports...\n")

modules = [
    ("pyclaw", "Main package"),
    ("pyclaw.core.message", "Message model"),
    ("pyclaw.core.session", "Session manager"),
    ("pyclaw.core.agent", "Agent core"),
    ("pyclaw.models.base", "Model base"),
    ("pyclaw.models.openai", "OpenAI provider"),
    ("pyclaw.tools.base", "Tool base"),
    ("pyclaw.tools.terminal", "Terminal tool"),
    ("pyclaw.tools.files", "File tools"),
    ("pyclaw.tools.registry", "Tool registry"),
    ("pyclaw.channels.base", "Channel base"),
    ("pyclaw.channels.telegram", "Telegram channel"),
    ("pyclaw.gateway.gateway", "Gateway"),
    ("pyclaw.infra.config", "Config"),
    ("pyclaw.cli.main", "CLI app"),
]

all_ok = True
for module, desc in modules:
    try:
        __import__(module)
        print(f"✅ {desc}")
    except Exception as e:
        print(f"❌ {desc}: {e}")
        all_ok = False

print("\n" + "="*50)
if all_ok:
    print("🎉 All imports successful!")
    print("\nNext steps:")
    print("  1. poetry install")
    print("  2. poetry run pyclaw init")
    print("  3. Edit config file")
    print("  4. poetry run pyclaw start")
else:
    print("⚠️ Some imports failed, please check errors")
    sys.exit(1)
