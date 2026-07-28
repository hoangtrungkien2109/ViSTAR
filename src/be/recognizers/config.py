"""Environment-backed recognizer configuration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(path: str | Path = ".env") -> bool:
        """Load simple KEY=VALUE entries when python-dotenv is unavailable."""
        env_path = Path(path)
        if not env_path.exists():
            return False
        for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key:
                os.environ.setdefault(key, value)
        return True


load_dotenv()


DEFAULT_ACTIONS = [
    "Xin chao",
    "Tu Choi",
    "Le Halloween",
    "Ruc Ro",
    "May Man",
    "Nhan Vien",
    "Dia Chi",
    "San Truong",
    "Thay",
    "Toi",
    "Khong quen",
    "Nghi Hoc",
    "Tiep tan",
    "Ngay nay",
    "Cam on",
    "Xin loi",
    "Ky nang",
    "Hap dan",
    "Thuong Xuyen",
]

DEFAULT_TTS_ACTIONS = [
    "Xin chào",
    "Từ chối",
    "Lễ Halloween",
    "Rực rỡ",
    "May mắn",
    "Nhân viên",
    "Địa chỉ",
    "Sân trường",
    "Thầy",
    "Tôi",
    "Không quên",
    "Nghỉ học",
    "Tiếp tân",
    "Ngày nay",
    "Cảm ơn",
    "Xin lỗi",
    "Kỹ năng",
    "Hấp dẫn",
    "Thường xuyên",
]


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _optional_path(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    return Path(value) if value else None


def _labels_from_file(path: Path) -> tuple[list[str], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Recognizer label file does not exist: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, list):
            actions = [str(item) for item in payload]
            return actions, actions
        if isinstance(payload, dict):
            actions = [str(item) for item in payload["actions"]]
            tts = [str(item) for item in payload.get("actions_tts", actions)]
            return actions, tts
        raise ValueError("Recognizer label JSON must be a list or an object.")
    actions = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return actions, actions


@dataclass(frozen=True)
class RecognizerSettings:
    mode: str
    labels: tuple[str, ...]
    tts_labels: tuple[str, ...]
    threshold: float
    sequence_length: int
    device: str
    baseline_checkpoint: Path
    felf_checkpoint: Path | None
    felf_baseline_checkpoint: Path | None
    felf_lrg_checkpoint: Path | None
    felf_rf_checkpoint: Path | None
    felf_mt_checkpoint: Path | None
    felf_require_mt: bool
    felf_lrg_weight: float
    felf_rf_weight: float
    felf_stage2_baseline_weight: float
    felf_stage2_felf_weight: float
    felf_stage2_mt_weight: float
    felf_branch_dim: int
    felf_num_layers: int
    felf_num_heads: int
    felf_ff_dim: int
    felf_dropout: float
    felf_mt_branch_dim: int
    felf_mt_conditioning: str
    felf_mt_feature_version: str

    @classmethod
    def from_env(cls) -> "RecognizerSettings":
        mode = os.getenv("SIGN_RECOGNIZER", "baseline").strip().lower()
        if mode not in {"baseline", "felf_slr"}:
            raise ValueError(
                "SIGN_RECOGNIZER must be 'baseline' or 'felf_slr', "
                f"got {mode!r}"
            )

        label_path = _optional_path("RECOGNIZER_LABELS_FILE")
        if label_path is None:
            mode_label_variable = (
                "BASELINE_LABELS_FILE"
                if mode == "baseline"
                else "FELF_LABELS_FILE"
            )
            label_path = _optional_path(mode_label_variable)
        if label_path is None:
            actions, tts_actions = DEFAULT_ACTIONS, DEFAULT_TTS_ACTIONS
        else:
            actions, tts_actions = _labels_from_file(label_path)
        if not actions:
            raise ValueError("At least one recognizer label is required.")
        if len(actions) != len(tts_actions):
            raise ValueError("actions and actions_tts must have the same length.")

        sequence_length = int(os.getenv("RECOGNIZER_SEQUENCE_LENGTH", "40"))
        if sequence_length != 40:
            raise ValueError(
                "The released Baseline and FELF-SLR checkpoints require exactly "
                "40 frames."
            )

        return cls(
            mode=mode,
            labels=tuple(actions),
            tts_labels=tuple(tts_actions),
            threshold=float(os.getenv("RECOGNIZER_THRESHOLD", "0.70")),
            sequence_length=sequence_length,
            device=os.getenv("RECOGNIZER_DEVICE", "auto").strip().lower(),
            baseline_checkpoint=Path(
                os.getenv("BASELINE_CHECKPOINT", "src/be/n2_dict.pth")
            ),
            felf_checkpoint=_optional_path("FELF_CHECKPOINT"),
            felf_baseline_checkpoint=_optional_path("FELF_BASELINE_CHECKPOINT"),
            felf_lrg_checkpoint=_optional_path("FELF_LRG_CHECKPOINT"),
            felf_rf_checkpoint=_optional_path("FELF_RF_CHECKPOINT"),
            felf_mt_checkpoint=_optional_path("FELF_MT_CHECKPOINT"),
            felf_require_mt=_bool_env("FELF_REQUIRE_MT", True),
            felf_lrg_weight=float(os.getenv("FELF_LRG_WEIGHT", "1.0")),
            felf_rf_weight=float(os.getenv("FELF_RF_WEIGHT", "0.75")),
            felf_stage2_baseline_weight=float(
                os.getenv("FELF_STAGE2_BASELINE_WEIGHT", "0.5")
            ),
            felf_stage2_felf_weight=float(
                os.getenv("FELF_STAGE2_FELF_WEIGHT", "1.0")
            ),
            felf_stage2_mt_weight=float(
                os.getenv("FELF_STAGE2_MT_WEIGHT", "0.5")
            ),
            felf_branch_dim=int(os.getenv("FELF_BRANCH_DIM", "96")),
            felf_num_layers=int(os.getenv("FELF_NUM_LAYERS", "2")),
            felf_num_heads=int(os.getenv("FELF_NUM_HEADS", "4")),
            felf_ff_dim=int(os.getenv("FELF_FF_DIM", "768")),
            felf_dropout=float(os.getenv("FELF_DROPOUT", "0.2")),
            felf_mt_branch_dim=int(os.getenv("FELF_MT_BRANCH_DIM", "128")),
            felf_mt_conditioning=os.getenv(
                "FELF_MT_CONDITIONING", "none"
            ).strip(),
            felf_mt_feature_version=os.getenv(
                "FELF_MT_FEATURE_VERSION", "v1"
            ).strip(),
        )

    def public_dict(self) -> dict[str, object]:
        output = {
            "mode": self.mode,
            "classes": len(self.labels),
            "sequence_length": self.sequence_length,
            "threshold": self.threshold,
            "device": self.device,
        }
        if self.mode == "felf_slr":
            output.update(
                {
                    "stage2_mt_required": self.felf_require_mt,
                    "stage2_mt_configured": self.felf_mt_checkpoint is not None,
                }
            )
        return output
