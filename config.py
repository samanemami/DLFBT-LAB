from pathlib import Path


def get_source_module(lab_number):
    return "source" if Path("source.py").exists() else f"dlfbt_lab{lab_number}"
