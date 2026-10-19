import json
from pathlib import Path
from datasets import load_dataset


def download_ds(
    dataset_name: str = "sayakpaul/coco-mini",
    split: str = "train",
    output_dir: str = "./data/hf_1k_sample",
    num_samples: int = 1000,
):
    base_dir = Path(output_dir)
    images_dir = base_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {num_samples} samples from Hugging Face ({dataset_name})...")
    
    # streaming=True streams items one by one without downloading the full dataset archive
    ds = load_dataset(dataset_name, split=split, streaming=True)

    metadata = []

    for idx, item in enumerate(ds):
        if idx >= num_samples:
            break

        # 'image' column holds a PIL Image object
        pil_img = item["image"]
        image_id = item.get("image_id", idx)
        filename = f"{image_id:012d}.jpg" if isinstance(image_id, int) else f"{image_id}.jpg"
        save_path = images_dir / filename

        # Save image directly to disk
        pil_img.save(save_path, format="JPEG")

        # Collect annotation metadata if present
        metadata.append({
            "image_id": image_id,
            "file_name": filename,
            "width": pil_img.width,
            "height": pil_img.height,
            "objects": item.get("objects", {}),
        })

        if (idx + 1) % 100 == 0 or (idx + 1) == num_samples:
            print(f"  [{idx + 1}/{num_samples}] Saved {filename}")

    # Save metadata summary
    with open(base_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Downloaded {num_samples} images to: {images_dir.resolve()}")


if __name__ == "__main__":
    download_ds(
        dataset_name="detection-datasets/coco",
        split="train",
        output_dir="./data/coco_1k_hf",
        num_samples=1000,
    )