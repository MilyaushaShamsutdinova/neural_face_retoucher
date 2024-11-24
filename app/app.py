import streamlit as st
from PIL import Image
import requests
import io

API_URL = "http://127.0.0.1:8000/retouch"

def upload_and_retouch_image():
    st.title("Neural Face Retoucher")

    st.write("Upload an image, and our AI will retouch it for you!")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Retouch Image"):
            st.write("Processing...")

            uploaded_file.seek(0)
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

            try:
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    retouched_image = Image.open(io.BytesIO(response.content))
                    st.image(retouched_image, caption="Retouched Image", use_container_width=True)
                else:
                    st.error(f"Failed to process the image. Error: {response.json().get('detail')}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    upload_and_retouch_image()
