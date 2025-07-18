import importlib
import logging

logger = logging.getLogger(__name__)


class AutotuneManager:
    _enabled = False

    @classmethod
    def enable_autotune(cls):
        cls._enabled = True
        cls._reload_all_modules()
        logger.info("Autotune enabled")

    @classmethod
    def disable_autotune(cls):
        cls._enabled = False
        cls._reload_all_modules()
        logger.info("Autotune disabled")

    @classmethod
    def is_enabled(cls):
        return cls._enabled

    @staticmethod
    def _reload_all_modules():
        from .minifloat.kernels import cast as minifloat_cast

        importlib.reload(minifloat_cast)
