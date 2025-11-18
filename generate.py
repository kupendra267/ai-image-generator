from diffusers import StableDiffusionPipeline
import torch

STYLE_PRESETS = {
    "default": "",
    "anime": "anime style, colorful, vibrant",
    "photorealistic": "realistic, detailed, 8k, high quality",
    "cartoon": "cartoon style, flat colors, bold outlines"
}

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    model_id = "gsdf/Counterfeit-V2.5"  # your small model
    device = get_device()

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None
    )
    pipe = pipe.to(device)
    return pipe

def build_prompt(prompt, style="default"):
    style_text = STYLE_PRESETS.get(style, "")
    if style_text:
        return f"{prompt}, {style_text}"
    return prompt

def generate_images(
    prompt,
    num_images=1,
    negative_prompt="",
    style="default",
    guidance_scale=6.0,
    num_inference_steps=20
):
    pipe = load_model()
    final_prompt = build_prompt(prompt, style)

    images = pipe(
        prompt=[final_prompt] * num_images,
        negative_prompt=[negative_prompt] * num_images if negative_prompt else None,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    ).images

    return images

