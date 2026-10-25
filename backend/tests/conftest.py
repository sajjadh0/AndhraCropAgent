"""Pytest configuration ensuring fast and reproducible offline test runs."""

import os
import sys
from pathlib import Path

# Set environment paths and offline flags before any app modules are imported
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

os.environ["CHRONOS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
