from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from geotraj_lc.tools.correct_dets import correct_detections


__all__ = ["correct_detections"]
