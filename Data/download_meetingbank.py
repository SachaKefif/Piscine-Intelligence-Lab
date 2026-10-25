from datasets import load_dataset
from pathlib import Path

# Download MeetingBank from Hugging Face
dataset = load_dataset("huuuyeah/meetingbank")

# Save inside Data/MeetingBank
output = Path(__file__).resolve().parent / "Data" / "MeetingBank"
output.mkdir(parents=True, exist_ok=True)

# Export all splits to JSONL
for split, data in dataset.items():
    path = output / f"{split}.jsonl"
    data.to_json(str(path), force_ascii=False)
    print(f"Saved {split}: {len(data)} records -> {path}")

    print("MeetingBank downloaded successfully!")
