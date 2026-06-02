from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Set, Tuple
import re

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SequencePaths:
    sequence: str
    corrected_tracks: Path
    lane_change_gt: Path
    homography: Path
    centerline: Path


def _extract_base_sequence(stem: str) -> str:
    match = re.match(r"^(.*?)_clip_\d+$", stem)
    if match:
        return match.group(1)
    return stem


@dataclass(frozen=True)
class DataLayout:
    root: Path
    data_root: Path
    raw_tracks_dir: Path
    corrected_tracks_dir: Path
    gt_states_dir: Path
    homography_dir: Path
    centerline_dir: Path

    def ensure_dirs(self) -> None:
        for path in [
            self.raw_tracks_dir,
            self.corrected_tracks_dir,
            self.gt_states_dir,
            self.homography_dir,
            self.centerline_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _stems(directory: Path, pattern: str) -> Set[str]:
        if not directory.exists():
            return set()
        return {p.stem for p in directory.glob(pattern) if p.is_file()}

    def available_sequences(self) -> List[str]:
        corrected_stems = self._stems(self.corrected_tracks_dir, "*.txt")
        gt_stems = self._stems(self.gt_states_dir, "*.txt")
        homo_stems = self._stems(self.homography_dir, "*.npy")
        centerline_stems = self._stems(self.centerline_dir, "*.json")

        clip_sequences = []
        for clip_stem in corrected_stems:
            if clip_stem in gt_stems:
                base_seq = _extract_base_sequence(clip_stem)
                if base_seq in homo_stems and base_seq in centerline_stems:
                    clip_sequences.append(clip_stem)

        return sorted(clip_sequences)

    def split_requested_sequences(self, requested: Sequence[str]) -> Tuple[List[str], List[str]]:
        available = set(self.available_sequences())
        selected: List[str] = []
        missing: List[str] = []
        for seq in requested:
            if seq in available:
                selected.append(seq)
            else:
                missing.append(seq)
        return selected, missing

    def sequence_paths(self, sequence: str) -> SequencePaths:
        base_seq = _extract_base_sequence(sequence)
        seq_paths = SequencePaths(
            sequence=sequence,
            corrected_tracks=self.corrected_tracks_dir / f"{sequence}.txt",
            lane_change_gt=self.gt_states_dir / f"{sequence}.txt",
            homography=self.homography_dir / f"{base_seq}.npy",
            centerline=self.centerline_dir / f"{base_seq}.json",
        )

        missing = [
            path
            for path in [
                seq_paths.corrected_tracks,
                seq_paths.lane_change_gt,
                seq_paths.homography,
                seq_paths.centerline,
            ]
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing files for sequence '{sequence}': " + ", ".join(str(p) for p in missing)
            )
        return seq_paths


@dataclass(frozen=True)
class ExperimentLayout:
    root: Path
    cache_dir: Path
    splits_dir: Path
    checkpoints_dir: Path
    best_checkpoint_dir: Path

    @property
    def splits_path(self) -> Path:
        return self.splits_dir / "dataset_splits.pkl"

    @property
    def best_model_path(self) -> Path:
        return self.best_checkpoint_dir / "tcn_model.pth"

    @property
    def best_normalizer_path(self) -> Path:
        return self.best_checkpoint_dir / "normalizer.pkl"

    def ensure_dirs(self) -> None:
        for path in [
            self.cache_dir,
            self.splits_dir,
            self.best_checkpoint_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def existing_best_model_path(self) -> Path:
        legacy_paths = [
            self.root / "checkpoints" / "best" / "tcn_model.pth",
            self.root / "best_model" / "tcn_model.pth",
            self.root / "configs" / "tcn_model.pth",
        ]
        for path in legacy_paths:
            if path.exists():
                return path
        return self.best_model_path

    def existing_best_normalizer_path(self) -> Path:
        legacy_paths = [
            self.root / "checkpoints" / "best" / "normalizer.pkl",
            self.root / "best_model" / "normalizer.pkl",
            self.root / "configs" / "normalizer.pkl",
        ]
        for path in legacy_paths:
            if path.exists():
                return path
        return self.best_normalizer_path


def _pick_existing_dir(candidates: List[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _sequence_count_for_root(data_root: Path) -> int:
    corrected_dir = _pick_existing_dir(
        [
            data_root / "tracks" / "mot_refined",
            data_root / "tracks" / "refined",
            data_root / "detections" / "corrected",
        ]
    )
    gt_dir = _pick_existing_dir(
        [
            data_root / "gt",
            data_root / "gt" / "lane_change_frame",
            data_root / "annotations" / "frame_labels",
            data_root / "annotations" / "lane_change_states",
        ]
    )
    homo_dir = data_root / "maps" / "homography"
    centerline_dir = data_root / "maps" / "centerlines"

    corrected_stems = {p.stem for p in corrected_dir.glob("*.txt") if p.is_file()} if corrected_dir.exists() else set()
    gt_stems = {p.stem for p in gt_dir.glob("*.txt") if p.is_file()} if gt_dir.exists() else set()
    homo_stems = {p.stem for p in homo_dir.glob("*.npy") if p.is_file()} if homo_dir.exists() else set()
    centerline_stems = {p.stem for p in centerline_dir.glob("*.json") if p.is_file()} if centerline_dir.exists() else set()

    count = 0
    for clip_stem in corrected_stems:
        if clip_stem in gt_stems:
            base_seq = _extract_base_sequence(clip_stem)
            if base_seq in homo_stems and base_seq in centerline_stems:
                count += 1

    return count


def _pick_data_root(root_path: Path) -> Path:
    candidates = [
        root_path / "RLC_Dataset",
        root_path / "datasets",
        root_path / "dataset",
        root_path / "data",
    ]

    existing = [c for c in candidates if c.exists()]
    if not existing:
        return candidates[0]

    scored = sorted(((_sequence_count_for_root(c), c) for c in existing), key=lambda x: x[0], reverse=True)
    if scored and scored[0][0] > 0:
        return scored[0][1]
    return existing[0]


def default_layout(root: str = None) -> DataLayout:
    root_path = Path(root).resolve() if root else PROJECT_ROOT
    data_root = _pick_data_root(root_path)

    raw_tracks_dir = _pick_existing_dir([
        data_root / "tracks" / "mot_raw",
        data_root / "tracks" / "raw",
        data_root / "detections" / "raw",
    ])
    corrected_tracks_dir = _pick_existing_dir([
        data_root / "tracks" / "refined",
        data_root / "tracks" / "mot_refined",
        data_root / "detections" / "corrected",
    ])
    gt_states_dir = _pick_existing_dir([
        data_root / "annotations" / "frame_labels",
        data_root / "gt",
        data_root / "gt" / "lane_change_frame",
        data_root / "annotations" / "lane_change_states",
    ])
    homography_dir = _pick_existing_dir([data_root / "maps" / "homography"])
    centerline_dir = _pick_existing_dir([data_root / "maps" / "centerlines"])

    return DataLayout(
        root=root_path,
        data_root=data_root,
        raw_tracks_dir=raw_tracks_dir,
        corrected_tracks_dir=corrected_tracks_dir,
        gt_states_dir=gt_states_dir,
        homography_dir=homography_dir,
        centerline_dir=centerline_dir,
    )


def default_experiment(root: str = None) -> ExperimentLayout:
    root_path = Path(root).resolve() if root else PROJECT_ROOT
    checkpoints_dir = root_path / "checkpoints"
    best_checkpoint_dir = checkpoints_dir / "best"
    return ExperimentLayout(
        root=root_path,
        cache_dir=root_path / "cache" / "features",
        splits_dir=root_path / "cache" / "splits",
        checkpoints_dir=checkpoints_dir,
        best_checkpoint_dir=best_checkpoint_dir,
    )


