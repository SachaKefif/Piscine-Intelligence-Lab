"""Download the pinned MeetingBank split files into Data/Datasets/MeetingBank.

Run: python Data/Processing/download_meetingbank.py
The revision is the one found in the local Hugging Face cache during the
28 October 2026 data audit. Existing local exports are changed only if this
script is run explicitly.
"""

from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Data" / "Datasets" / "MeetingBank"
DATASET_ID = "huuuyeah/meetingbank"
REVISION = "5b4fb6f67f490be93e249d70b732780932d19fe3"


def main():
    dataset = load_dataset(DATASET_ID, revision=REVISION)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for split, data in dataset.items():
        path = OUTPUT / f"{split}.jsonl"
        data.to_json(str(path), force_ascii=False)
        print(f"Saved {split}: {len(data)} rows -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
