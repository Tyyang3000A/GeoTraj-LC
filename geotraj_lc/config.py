from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import re


@dataclass
class Config:
    fps: float = 20.0
    max_gap_time: float = 1.0
    process_noise_q: float = 0.001
    measurement_noise_r: float = 0.001

    scale_overrides: Dict[str, float] = field(
        default_factory=lambda: {
            "G0512-K111": 20.0,
            "G4202-K32": 9.5,
            "G4215-K12": 9.17,
            "S2-K152": 5.17,
            "S2-K188_0": 3.5,
            "S2-K188_1": 3.5,
            "S2-K201": 6.0,
            "S3-K316": 6.83,
        }
    )

    train_prefixes: Tuple[str, ...] = ("G4202-K32", "S2-K188_0", "S2-K188_1", "S3-K316")
    val_prefixes: Tuple[str, ...] = ("S2-K152", "S2-K201")
    test_prefixes: Tuple[str, ...] = ("G0512-K111", "G4215-K12")

    # TCN model
    tcn_input_size: int = 12
    tcn_output_size: int = 2
    tcn_num_channels: Tuple[int, ...] = (32, 32, 64, 64)
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.3

    # TCN detector
    tcn_confidence_threshold: float = 0.87
    tcn_min_trigger_frames: int = 3
    tcn_feature_dim: int = 12

    # Training
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 0.0005
    weight_decay: float = 5e-4
    positive_weight_factor: float = 0.2
    grad_clip_max_norm: float = 1.0
    train_event_eval_interval: int = 5

    @property
    def dt(self) -> float:
        return 1.0 / self.fps

    @property
    def max_gap_frames(self) -> int:
        return int(round(self.max_gap_time / self.dt))

    def _extract_base_sequence(self, sequence: str) -> str:
        match = re.match(r"^(.*?)_clip_\d+$", sequence)
        if match:
            return match.group(1)
        return sequence

    def scale_for_sequence(self, sequence: str) -> float:
        base_seq = self._extract_base_sequence(sequence)
        return float(self.scale_overrides.get(base_seq, 20.0))


def get_data_splits(layout, config: Config = None) -> Tuple[List[str], List[str], List[str]]:
    if config is None:
        config = Config()
    available = layout.available_sequences()
    train_seqs = [s for s in available if any(p in s for p in config.train_prefixes)]
    val_seqs = [s for s in available if any(p in s for p in config.val_prefixes)]
    test_seqs = [s for s in available if any(p in s for p in config.test_prefixes)]
    return train_seqs, val_seqs, test_seqs

