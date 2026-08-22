import sys
from pathlib import Path

from loguru import logger


_LOG_CONFIGURADO = False

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
ARQUIVO_LOG = LOG_DIR / "itau_recuperaca_pj.log"


def setup_logs_once():
    global _LOG_CONFIGURADO

    if _LOG_CONFIGURADO:
        return logger

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger.remove()

    # ---------------------------------------------------------
    # Console
    # ---------------------------------------------------------
    logger.add(
        sys.stderr,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level="INFO",
    )

    # ---------------------------------------------------------
    # Arquivo
    # ---------------------------------------------------------
    logger.add(
        ARQUIVO_LOG,
        rotation="5 MB",
        retention="7 days",
        encoding="utf-8",
        level="DEBUG",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{function}:{line} - {message}"
        ),
    )

    _LOG_CONFIGURADO = True

    logger.info("Sistema de logs inicializado")

    return logger