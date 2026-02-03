from diffusers import StableDiffusionPipeline
import torch
import streamlit as st

STYLE_PRESETS = {
    "default": "",
    "anime": "anime style, colorful, vibrant",
    "photorealistic": "realistic, detailed, 8k, high quality",
    "cartoon": "cartoon style, flat colors, bold outlines"
}

MODEL_ID = "gsdf/Counterfeit-V2.5"

# ✅ Load model ONCE (very important)
@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        safety_checker=None
    )
    pipe = pipe.to("cpu")                 # ✅ force CPU
    pipe.enable_attention_slicing()        # ✅ reduce RAM
    return pipe


def build_prompt(prompt, style="default"):
    style_text = STYLE_PRESETS.get(style, "")
    if style_text:
        return f"{prompt}, {style_text}"
    return prompt


def generate_images(
    prompt,
    negative_prompt="",
    style="default",
    guidance_scale=6.0,
    num_inference_steps=15
):
    pipe = load_model()
    final_prompt = build_prompt(prompt, style)

    result = pipe(
        prompt=final_prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=512,
        width=512
    )

    return result.images


