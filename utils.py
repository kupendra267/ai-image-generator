from pathlib import Path
import json
import time

# Folders to store images and metadata
SAMPLES_DIR = Path("samples")
META_DIR = Path("metadata")

# Create folders if they don't exist
SAMPLES_DIR.mkdir(exist_ok=True)
META_DIR.mkdir(exist_ok=True)

def save_images_with_metadata(
    images,
    prompt,
    negative_prompt="",
    style="default",
    guidance_scale=6.0,
    num_inference_steps=20
):
    """
    Save list of PIL images to 'samples/' and write a JSON metadata file
    to 'metadata/' folder.
    """
    timestamp = int(time.time())
    image_paths = []

    # Save each image with a unique name
    for idx, img in enumerate(images, start=1):
        filename = f"img_{timestamp}_{idx}.png"
        path = SAMPLES_DIR / filename
        img.save(path)
        image_paths.append(str(path))

    # Build metadata dictionary
    meta = {
        "timestamp": timestamp,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "style": style,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "images": image_paths
    }

    # Save metadata JSON
    meta_file = META_DIR / f"meta_{timestamp}.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    return image_paths, str(meta_file)
