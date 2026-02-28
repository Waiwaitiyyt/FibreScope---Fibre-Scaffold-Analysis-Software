"""
程序启动脚本
"""
import sys
from pathlib import Path

# 添加src/python到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src" / "python"))

# 导入并运行主程序
from src.python.main import main

if __name__ == "__main__":
    main()