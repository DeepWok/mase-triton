import importlib


class AutotuneManager:
    _enabled = False

    @classmethod
    def enable(cls):
        cls._enabled = True
        cls._reload_all_modules()

    @classmethod
    def disable(cls):
        cls._enabled = False
        cls._reload_all_modules()

    @classmethod
    def is_enabled(cls):
        return cls._enabled

    @staticmethod
    def _reload_all_modules():
        from . import minifloat

        importlib.reload(minifloat)
