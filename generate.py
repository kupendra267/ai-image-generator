import requests
from PIL import Image
from io import BytesIO
import streamlit as st

STYLE_PRESETS = {
    "default": "",
    "anime": "anime style, colorful, vibrant",
    "photorealistic": "realistic, detailed, high quality",
    "cartoon": "cartoon style, flat colors, bold outlines"
}

# ✅ Public Stable Diffusion Space (no auth required)
API_URL = "https://hf.space/embed/stabilityai/stable-diffusion/+/api/predict"

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
        "data": [
            final_prompt,          # prompt
            negative_prompt,       # negative prompt
            num_inference_steps,   # steps
            guidance_scale         # guidance scale
        ]
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code != 200:
        st.error(f"Space API error: {response.text}")
        return []

    try:
        result = response.json()
        image_url = result["data"][0]
        image_bytes = requests.get(image_url).content
        image = Image.open(BytesIO(image_bytes))
        return [image]
    except Exception as e:
        st.error(f"Failed to process image: {e}")
        return []
