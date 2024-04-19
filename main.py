import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "upload"

@app.get("/")
async def hello():
    try:
        return JSONResponse(content={"message": "hello from Fast API"})
    except Exception as e:
        return JSONResponse(content={"message": "error", "error": str(e)})

@app.post("/predict")
async def predict(user_image: UploadFile = File(...)):
    try:
        # Save the uploaded image file
        contents = await user_image.read()
        filename = os.path.join(UPLOAD_FOLDER, user_image.filename)
        with open(filename, "wb") as f:
            f.write(contents)
        # Send the image to the model for prediction
        # For now, we'll just return success
        print("Image Uploaded Successfully!")
        return JSONResponse(content={"message": "success"})
    except Exception as e:
        # If an error occurs, return error message
        return JSONResponse(content={"message": "error", "error": str(e)})


# To start the server write: uvicorn main:app --reload




