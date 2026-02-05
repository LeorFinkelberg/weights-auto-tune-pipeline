import sys
from loguru import logger

FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function} | {message}"


def setup_logging():
    if logger._core.handlers:
        return
    logger.remove()

    logger.add(
        sys.stderr,
        level="INFO",
        format=FORMAT,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    logger.add(
        "logs/app_{time:YYYY-MM-DD}.log",
        level="INFO",
        format=FORMAT,
        rotation="50MB",
        retention="30 days",
        compression="zip",
        serialize=True,
    )

    logger.info("Logging configured successfully")
