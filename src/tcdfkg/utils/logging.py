import logging, os, sys
from logging import StreamHandler, FileHandler
from typing import Optional

def get_logger(name: str = "tcdfkg", level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    lvl = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(lvl)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = StreamHandler(sys.stdout); sh.setFormatter(fmt); sh.setLevel(lvl)
    logger.addHandler(sh)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = FileHandler(log_file); fh.setFormatter(fmt); fh.setLevel(lvl)
        logger.addHandler(fh)
    logger.propagate = False
    return logger
