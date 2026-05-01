"""
Home.py  —  Root-level entry point for Streamlit Cloud deployment.
Streamlit Cloud looks for a .py file at the repo root.
This file simply re-runs the main app from the app/ subfolder.
"""
import runpy, os, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

# Run the actual app file
runpy.run_path(str(REPO_ROOT / "app" / "streamlit_app.py"), run_name="__main__")
