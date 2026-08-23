"""
统一日志模块 — 控制台 + JSONL 文件双输出

用法:
    from utils.log import logger
    logger.info("something happened")
    logger.warning("watch out")
    logger.error("something broke")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str = "aero") -> logging.Logger:
    """创建/获取命名 logger，自动配置控制台 + JSONL 文件输出"""
    lg = logging.getLogger(name)

    if lg.handlers:
        return lg

    lg.setLevel(logging.INFO)
    lg.propagate = False

    # 控制台 handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    lg.addHandler(ch)

    # 文件 handler (JSONL 格式)
    log_dir = Path(__file__).resolve().parent.parent / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_dir / "app.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        '{"time":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","msg":"%(message)s"}',
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))
    lg.addHandler(fh)

    return lg


logger = get_logger()
