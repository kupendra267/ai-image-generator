import streamlit as st
from generate import generate_images, STYLE_PRESETS
from utils import save_images_with_metadata

st.set_page_config(page_title="AI Image Generator", layout="wide")

st.title("🖼️ AI-Powered Text-to-Image Generator")

st.write(
    "Enter a text prompt below and generate images using an open-source model. "
    "Note: On CPU this can take 30–90 seconds."
)

# --- MAIN INPUTS ---
prompt = st.text_area(
    "Text prompt",
    value="a cute anime girl smiling",
    height=80
)

negative_prompt = st.text_input(
    "Negative prompt (optional)",
    value="low quality, blurry"
)

col1, col2, col3 = st.columns(3)

with col1:
    style = st.selectbox(
        "Style",
        options=list(STYLE_PRESETS.keys()),
        index=0
    )

with col2:
    num_images = st.slider("Number of images", min_value=1, max_value=4, value=1)

with col3:
    guidance_scale = st.slider("Guidance scale", 1.0, 12.0, 6.0)

num_steps = st.slider("Number of inference steps", 10, 40, 20)

st.markdown(
    f"**Estimated time on CPU:** ~{num_images * num_steps // 10 + 10}–"
    f"{num_images * num_steps // 10 + 40} seconds"
)

generate_btn = st.button("🚀 Generate Images")

# --- GENERATION ---
if generate_btn:
    if not prompt.strip():
        st.error("Please enter a prompt.")
    else:
        with st.spinner("Generating images... This may take a while on CPU."):
            images = generate_images(
                prompt=prompt,
                num_images=num_images,
                negative_prompt=negative_prompt,
                style=style,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps
            )

            paths, meta_path = save_images_with_metadata(
                images,
                prompt=prompt,
                negative_prompt=negative_prompt,
                style=style,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps
            )

        st.success(f"Done! Metadata saved at: `{meta_path}`")

        st.subheader("Generated Images")
        cols = st.columns(min(4, num_images))

        for i, (img, path) in enumerate(zip(images, paths)):
            with cols[i % len(cols)]:
                st.image(img, caption=path, use_column_width=True)
                with open(path, "rb") as f:
                    st.download_button(
                        label="Download",
                        data=f,
                        file_name=path.split("/")[-1],
                        mime="image/png",
                        key=f"download_{i}"
                    )
