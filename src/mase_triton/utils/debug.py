import os
import logging

from .deps import package_is_available

logger = logging.getLogger(__name__)


def set_ipdb_breakpoint():
    if not package_is_available("ipdb"):
        raise ImportError("ipdb is not available, please install it via pip install ipdb")
    os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
    logger.info("Set ipdb as default breakpoint")
