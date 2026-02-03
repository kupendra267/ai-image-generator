import requests
from PIL import Image
from io import BytesIO
import os
import streamlit as st

STYLE_PRESETS = {
    "default": "",
    "anime": "anime style, colorful, vibrant",
    "photorealistic": "realistic, detailed, high quality",
    "cartoon": "cartoon style, flat colors, bold outlines"
}

API_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-2"

HEADERS = {
    "Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}",
    "Accept": "image/png"
}

def build_prompt(prompt, style):
    style_text = STYLE_PRESETS.get(style, "")
    return f"{prompt}, {style_text}" if style_text else prompt


def generate_images(
    prompt,
    negative_prompt="",
    style="default",
    guidance_scale=7.5,
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

    response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)

    # ❌ HTTP error (401, 403, 429, 503)
    if response.status_code != 200:
        st.error(f"Hugging Face API error {response.status_code}: {response.text}")
        return []

    content_type = response.headers.get("content-type", "")

    # ❌ HF returned JSON instead of image
    if "application/json" in content_type:
        st.warning("Model is loading or request was rejected. Please try again in 20–30 seconds.")
        try:
            st.json(response.json())
        except Exception:
            pass
        return []

    # ❌ HF returned something that is not an image
    if "image" not in content_type:
        st.error("Received non-image response from Hugging Face.")
        return []

    # ✅ Safe image load
    try:
        image = Image.open(BytesIO(response.content))
        return [image]
    except Exception as e:
        st.error(f"Failed to decode image: {e}")
        return []
