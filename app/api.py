from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image
from model.inference import FaceRetoucher


app = FastAPI()
face_retoucher = FaceRetoucher("model\weights\generator.pth")


@app.post("/retouch")
async def retouch_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    
    retouched_image = face_retoucher.retouch(image)

    output_buffer = BytesIO()
    retouched_image.save(output_buffer, format="JPEG")
    output_buffer.seek(0)

    return StreamingResponse(output_buffer, media_type="image/jpeg")
