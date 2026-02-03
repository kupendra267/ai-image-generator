from gradio_client import Client
from PIL import Image
import streamlit as st

STYLE_PRESETS = {
    "default": "",
    "anime": "anime style, colorful, vibrant",
    "photorealistic": "realistic, detailed, high quality",
    "cartoon": "cartoon style, flat colors, bold outlines"
}

# ✅ REAL, PUBLIC, WORKING SPACE
client = Client("runwayml/stable-diffusion-v1-5")


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

    try:
        result = client.predict(
            final_prompt,          # prompt
            negative_prompt,       # negative prompt
            num_inference_steps,   # steps
            guidance_scale,        # guidance scale
            api_name="/predict"
        )

        # result is image path
        image = Image.open(result)
        return [image]

    except Exception as e:
        st.error(f"Image generation failed: {e}")
        return []
