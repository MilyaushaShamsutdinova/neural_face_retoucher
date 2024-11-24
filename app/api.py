from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from model.inference import FaceRetoucher


app = FastAPI()
face_retoucher = FaceRetoucher("model\weights\generator.pth")


@app.post("/retouch")
async def retouch_image(file: UploadFile = File(...)):
    try:
        # check that file provided
        if not file:
            raise HTTPException(status_code=400, detail="No file provided.")

        # check file type
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPG or PNG image.")
        
        try:
            contents = await file.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read file. Error: {e}")

        try:
            image = Image.open(BytesIO(contents)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

        # inference
        try:
            retouched_image = face_retoucher.retouch_image(image)
            if not isinstance(retouched_image, Image.Image):
                raise ValueError("The retouch method did not return a valid PIL.Image object.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model inference failed. Error: {e}")

        output_buffer = BytesIO()
        retouched_image.save(output_buffer, format="JPEG")
        output_buffer.seek(0)

        return Response(content=output_buffer.getvalue(), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
