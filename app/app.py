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
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Retouch Image"):
            st.write("Processing...")

            response = requests.post(
                API_URL, 
                files={"file": uploaded_file}
            )

            if response.status_code == 200:
                retouched_image = Image.open(io.BytesIO(response.content))
                st.image(retouched_image, caption="Retouched Image", use_column_width=True)
            else:
                st.error("Failed to process the image. Please try again.")

if __name__ == "__main__":
    upload_and_retouch_image()
