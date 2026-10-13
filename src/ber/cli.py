"""Command-line options that all step scripts share. This module must not import other ber modules:
the options become environment variables, and ber/paths.py reads those variables at import time."""
from __future__ import annotations

import argparse
import os


def parse(description: str, extra=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", help="folder that contains train/ and test/ (default: ./dataset next to code/ or in this folder)")
    ap.add_argument("--work-dir", help="folder for intermediate files (default: code/business_entity_resolution/work)")
    ap.add_argument("--out-dir", help="folder for the two submission TSV files (default: output/ next to code/)")
    ap.add_argument("--threads", type=int, help="CPU threads (default: all CPUs available to the process)")
    if extra:
        extra(ap)
    args = ap.parse_args()
    for opt, var in (("data_dir", "BER_DATA_DIR"), ("work_dir", "BER_WORK_DIR"), ("out_dir", "BER_OUT_DIR"), ("threads", "BER_THREADS")):
        if getattr(args, opt) is not None:
            os.environ[var] = str(getattr(args, opt))
    return args
