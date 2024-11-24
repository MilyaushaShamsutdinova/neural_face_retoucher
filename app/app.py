import streamlit as st
from PIL import Image
import requests
import io


API_URL = "http://127.0.0.1:8000/retouch"

def upload_and_retouch_image():
    st.set_page_config(layout="wide")
    st.title("Neural Face Retoucher")
    st.write("Upload an image to retouch it!")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    st.markdown("---")

    col1, col2, _ = st.columns(3, gap="large")

    retouched_image_placeholder = None

    with col1:
        st.header("Original Image")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("Retouch Image"):
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

                try:
                    response = requests.post(API_URL, files=files)

                    if response.status_code == 200:
                        retouched_image = Image.open(io.BytesIO(response.content))
                        retouched_image_placeholder = retouched_image
                    else:
                        st.error(f"Failed to process the image. Error: {response.json().get('detail')}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.info("Please upload an image to see it here.")

    with col2:
        st.header("Retouched Image")
        if retouched_image_placeholder:
            st.image(retouched_image_placeholder, caption="Retouched Image", use_container_width=True)
        else:
            st.info("Retouched image will appear here after retouching.")

if __name__ == "__main__":
    upload_and_retouch_image()
