# Topodim/utils/log.py

import logging
import sys

# 1. 定义日志器的名称
# 通常使用应用的名称或模块名称
LOGGER_NAME = "Topodim"

# 2. 创建日志器实例
logger = logging.getLogger(LOGGER_NAME)

# 3. 设置日志级别
# DEBUG, INFO, WARNING, ERROR, CRITICAL
logger.setLevel(logging.INFO) 
# 在实际开发中，你可能希望在开发环境设置为 DEBUG

# 4. 防止重复添加处理器 (如果这个文件被多次导入)
if not logger.handlers:
    # 5. 创建一个处理器 (Handler) - 这里使用 StreamHandler 输出到控制台
    handler = logging.StreamHandler(sys.stdout)

    # 6. 设置处理器的级别
    handler.setLevel(logging.DEBUG)

    # 7. 定义日志的格式 (Formatter)
    # 格式包括时间、级别、日志名和消息
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 8. 将格式器添加到处理器
    handler.setFormatter(formatter)

    # 9. 将处理器添加到日志器
    logger.addHandler(handler)

# 示例：你可以通过以下方式使用这个 logger：
# logger.info("Logger initialized successfully.")
# logger.warning("A warning message.")