from loguru import logger

FORMAT_FILE = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function} | {message}"


def setup_logging():
    logger.add(
        "logs/app_{time:YYYY-MM-DD}.log",
        level="INFO",
        format=FORMAT_FILE,
        rotation="500MB",
        retention=False,
        enqueue=True,
        backtrace=True,
        diagnose=True,
        serialize=True,
    )

    logger.info("Logging configured successfully")
