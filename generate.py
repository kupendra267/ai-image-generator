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

# ✅ CORRECT ROUTER MODEL (WORKING)
API_URL = "https://router.huggingface.co/hf-inference/models/runwayml/stable-diffusion-v1-5"

# ✅ STREAMLIT SECRETS (CRITICAL FIX)
HF_TOKEN = st.secrets["HF_API_TOKEN"]

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Accept": "image/png",
    "Content-Type": "application/json"
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
    payload = {
        "inputs": build_prompt(prompt, style),
        "parameters": {
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps
        }
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120)

    # ❌ API error
    if response.status_code != 200:
        st.error(f"Hugging Face error {response.status_code}: {response.text}")
        return []

    # ⏳ Model loading response
    if "application/json" in response.headers.get("content-type", ""):
        st.warning("Model is loading. Wait 20–30 seconds and click Generate again.")
        return []

    try:
        image = Image.open(BytesIO(response.content))
        return [image]
    except Exception as e:
        st.error(f"Image decode failed: {e}")
        return []
