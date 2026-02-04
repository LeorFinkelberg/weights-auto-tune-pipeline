import sys
import os
from loguru import logger


DEV_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)

PROD_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function} | {message}"


def setup_logging():
    logger.remove()

    if os.getenv("ENV_TYPE", "dev") == "dev":
        logger.add(sys.stderr, level="DEBUG", format=DEV_FORMAT, colorize=True)
    else:
        logger.add(sys.stderr, level="INFO", format=PROD_FORMAT, colorize=False)

        logger.add(
            "logs/app.log",
            level="DEBUG",
            rotation="10 MB",
            retention="1 month",
            compression="zip",
            serialize=True,
        )
