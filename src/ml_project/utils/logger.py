import logging
from pathlib import Path


def setup_logger(
        log_file_path: Path,
        name: str,
        level: int = logging.DEBUG
) -> logging.Logger:

    logger = logging.getLogger(name)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        logger.setLevel(level)

        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        handler = logging.FileHandler(log_file_path, encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

