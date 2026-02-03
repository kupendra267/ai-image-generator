import streamlit as st
from generate import generate_images, STYLE_PRESETS

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI Image Generator",
    layout="wide"
)

# ------------------ TITLE ------------------
st.title("🖼️ AI-Powered Text-to-Image Generator")

st.write(
    "Generate images from text prompts using an AI model. "
    "This app uses a cloud-based inference API, so the first request may take a little longer."
)

# ------------------ INPUTS ------------------
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
    style = st.selectbox(
        "Style",
        options=list(STYLE_PRESETS.keys()),
        index=0
    )

with col2:
    guidance_scale = st.slider(
        "Guidance Scale",
        min_value=1.0,
        max_value=12.0,
        value=7.5
    )

num_steps = st.slider(
    "Inference Steps",
    min_value=10,
    max_value=30,
    value=20
)

st.markdown(
    f"**Note:** First generation may take ~20–30 seconds due to model loading."
)

# ------------------ GENERATE BUTTON ------------------
generate_btn = st.button("🚀 Generate Image")

# ------------------ GENERATION ------------------
if generate_btn:
    if not prompt.strip():
        st.error("Please enter a text prompt.")
        st.stop()

    with st.spinner("Generating image… Please wait."):
        images = generate_images(
            prompt=prompt,
            negative_prompt=negative_prompt,
            style=style,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps
        )

    # 🚨 If API returned error / loading response
    if not images:
        st.info("If this is your first request, please wait 20–30 seconds and try again.")
        st.stop()

    # ------------------ DISPLAY RESULT ------------------
    st.success("Image generated successfully!")

    image = images[0]
    st.image(image, caption="Generated Image", use_column_width=True)

    # ------------------ DOWNLOAD BUTTON ------------------
    import io
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")

    st.download_button(
        label="⬇️ Download Image",
        data=img_bytes.getvalue(),
        file_name="generated_image.png",
        mime="image/png"
    )

# ------------------ FOOTER ------------------
st.markdown("---")
st.caption(
    "Built with Streamlit & Hugging Face Inference API • "
    "Deployed on Streamlit Cloud"
)
