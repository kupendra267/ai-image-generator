import streamlit as st
from generate import generate_images, STYLE_PRESETS

st.set_page_config(page_title="AI Image Generator", layout="wide")

st.title("🖼️ AI-Powered Text-to-Image Generator")

prompt = st.text_area(
    "Text prompt",
    value="a cute anime girl smiling",
    height=80
)

negative_prompt = st.text_input(
    "Negative prompt (optional)",
    value="low quality, blurry"
)

style = st.selectbox(
    "Style",
    options=list(STYLE_PRESETS.keys())
)

guidance_scale = st.slider("Guidance scale", 1.0, 12.0, 7.5)
num_steps = st.slider("Inference steps", 10, 30, 20)

if st.button("🚀 Generate Image"):
    if not prompt.strip():
        st.error("Please enter a prompt")
    else:
        with st.spinner("Generating image..."):
            images = generate_images(
                prompt=prompt,
                negative_prompt=negative_prompt,
                style=style,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps
            )

        st.image(images[0], caption="Generated Image", use_column_width=True)
