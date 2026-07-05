"""允许通过 python3 -m trading_system <module> 运行子模块"""
from __future__ import annotations

import sys
from pathlib import Path

# 确保项目根目录在 sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 默认运行 cli
from trading_system.cli import main

if __name__ == "__main__":
    main()
