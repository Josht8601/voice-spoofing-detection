from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np

from preprocess import encode_label, preprocess_waveform


class ASVspoofDataLoader:
    def __init__(self, data_dir: str | Path, split: str = "train", allowed_systems: List[str] | None = None) -> None:
        self.data_dir = Path(data_dir)

        split_map = {
            "train": {
                "audio_dir": self.data_dir / "ASVspoof2019_LA_train" / "flac",
                "protocol_file": self.data_dir
                / "ASVspoof2019_LA_cm_protocols"
                / "ASVspoof2019.LA.cm.train.trn.txt",
            },
            "dev": {
                "audio_dir": self.data_dir / "ASVspoof2019_LA_dev" / "flac",
                "protocol_file": self.data_dir
                / "ASVspoof2019_LA_cm_protocols"
                / "ASVspoof2019.LA.cm.dev.trl.txt",
            },
        }

        if split not in split_map:
            raise ValueError(f"Unsupported split: {split}. Use 'train' or 'dev'.")

        self.split = split
        self.audio_dir = split_map[split]["audio_dir"]
        self.protocol_file = split_map[split]["protocol_file"]

        self.file_labels = self._load_protocol()

        if allowed_systems is not None:
            filtered = {}

            for file_name, info in self.file_labels.items():
                label = info["label"]
                system = info["system"]

                # ✅ ALWAYS keep bonafide
                if label == "bonafide":
                    filtered[file_name] = info

                # ✅ Keep ONLY selected spoof systems
                elif label == "spoof" and system in allowed_systems:
                    filtered[file_name] = info

            self.file_labels = filtered

        # DEBUG PRINT (temporary)
        systems = set()

        for info in self.file_labels.values():
            if info["system"] is not None:
                systems.add(info["system"])

        print("AVAILABLE SYSTEMS:", sorted(systems))

        self.file_names = list(self.file_labels.keys())

    def _load_protocol(self) -> Dict[str, dict]:
        if not self.protocol_file.exists():
            raise FileNotFoundError(f"Protocol file not found: {self.protocol_file}")

        file_labels: Dict[str, dict] = {}

        with open(self.protocol_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                file_name = parts[1]
                label = parts[-1]

                # 🔥 Handle bonafide vs spoof correctly
                if label == "bonafide":
                    system_id = None
                else:
                    system_id = parts[3]  # A01, A02, ...

                file_labels[file_name] = {
                    "label": label,
                    "system": system_id
                }

        return file_labels

    def __len__(self) -> int:
        return len(self.file_names)

    def load_audio(self, file_name: str) -> tuple[np.ndarray, int]:
        audio_path = self.audio_dir / f"{file_name}.flac"
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        waveform, sample_rate = librosa.load(audio_path, sr=None)
        return waveform, sample_rate

    def get_example(self, index: int) -> dict:
        file_name = self.file_names[index]
        #label_str = self.file_labels[file_name]
        info = self.file_labels[file_name]
        label_str = info["label"]
        system = info["system"]

        waveform, sample_rate = self.load_audio(file_name)

        processed_waveform = preprocess_waveform(waveform)
        label = encode_label(label_str)

        return {
            "file_name": file_name,
            "waveform": processed_waveform,
            "sample_rate": sample_rate,
            "label": label,
            "label_str": label_str,
            "system": system
        }

    def summary(self, n: int = 5) -> List[dict]:
        samples = []
        for i in range(min(n, len(self))):
            item = self.get_example(i)
            samples.append(
                {
                    "file_name": item["file_name"],
                    "shape": item["waveform"].shape,
                    "sample_rate": item["sample_rate"],
                    "label": item["label"],
                    "label_str": item["label_str"],
                }
            )
        return samples


if __name__ == "__main__":
    data_dir = Path("../data/LA")
    loader = ASVspoofDataLoader(data_dir=data_dir, split="train")

    print("Total examples:", len(loader))
    for item in loader.summary(3):
        print(item)