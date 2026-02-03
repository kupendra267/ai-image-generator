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

# ✅ NEW Hugging Face Router endpoint (IMPORTANT)
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

    response = requests.post(API_URL, headers=HEADERS, json=payload)

    # If HF returns JSON (error / loading)
    if response.headers.get("content-type", "").startswith("application/json"):
        st.error(response.json())
        return []

    image = Image.open(BytesIO(response.content))
    return [image]


