import requests
from PIL import Image
from io import BytesIO
import os

# Optional style presets (same as before)
STYLE_PRESETS = {
    "default": "",
    "anime": "anime style, colorful, vibrant",
    "photorealistic": "realistic, detailed, high quality",
    "cartoon": "cartoon style, flat colors, bold outlines"
}

# Hugging Face Inference API
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
HEADERS = {
    "Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"
}

def build_prompt(prompt, style="default"):
    style_text = STYLE_PRESETS.get(style, "")
    return f"{prompt}, {style_text}" if style_text else prompt


def generate_images(
    prompt,
    num_images=1,                 # kept for compatibility (ignored internally)
    negative_prompt="",
    style="default",
    guidance_scale=6.0,
    num_inference_steps=20
):
    final_prompt = build_prompt(prompt, style)

    payload = {
        "inputs": final_prompt,
        "parameters": {
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps
        }
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    response.raise_for_status()

    image = Image.open(BytesIO(response.content))

    # return list to match your existing app code
    return [image]
