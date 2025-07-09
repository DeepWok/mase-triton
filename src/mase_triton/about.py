from pathlib import Path

PACKAGE_NAME = "mase_triton"
__version__ = Path(__file__).with_name("VERSION").read_text().strip()
