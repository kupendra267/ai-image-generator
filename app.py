import streamlit as st
from generate import generate_images, STYLE_PRESETS
import io

st.set_page_config(page_title="AI Image Generator", layout="wide")

st.title("🖼️ AI Image Generator (Stable Diffusion)")

st.write(
    "Generate images using Hugging Face Stable Diffusion. "
    "First generation may take 20–30 seconds."
)

prompt = st.text_area(
    "Text Prompt",
    value="a cute anime girl smiling",
    height=80
)

negative_prompt = st.text_input(
    "Negative Prompt (optional)",
    value="low quality, blurry"
)

col1, col2 = st.columns(2)

with col1:
    style = st.selectbox("Style", list(STYLE_PRESETS.keys()))

with col2:
    guidance_scale = st.slider("Guidance Scale", 1.0, 12.0, 7.5)

num_steps = st.slider("Inference Steps", 10, 30, 20)

# 🔍 Debug (remove later if you want)
st.caption(f"HF Token loaded: {bool(st.secrets.get('HF_API_TOKEN'))}")

if st.button("🚀 Generate Image"):
    if not prompt.strip():
        st.error("Enter a prompt")
        st.stop()

    with st.spinner("Generating image…"):
        images = generate_images(
            prompt=prompt,
            negative_prompt=negative_prompt,
            style=style,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps
        )

    if not images:
        st.info("If this is first run, wait 30 seconds and try again.")
        st.stop()

    img = images[0]
    st.image(img, use_column_width=True)

    buf = io.BytesIO()
    img.save(buf, format="PNG")

    st.download_button(
        "⬇️ Download Image",
        buf.getvalue(),
        "generated.png",
        "image/png"
    )
