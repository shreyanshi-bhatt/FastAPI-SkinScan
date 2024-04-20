# Dermnet dataset used
# Notebook saved on Kaggle

# HAM model accuracy-test: 28/28 - 0s - loss: 0.3513 - accuracy: 0.8584 - 418ms/epoch - 15ms/step
# HAM model accuracy-train: 6s 255ms/step - loss: 0.1118 - accuracy: 0.9589 - val_loss: 0.4453 - val_accuracy: 0.8429
# HAM:
# # 7 diseases classify kare che
# from google.colab.patches import cv2_imshow
# import cv2
#
# classes = {4: ('nv', ' melanocytic nevi'), 6: ('mel', 'melanoma'), 2 :('bkl', 'benign keratosis-like lesions'),
# 1:('bcc' , ' basal cell carcinoma'), 5: ('vasc', ' pyogenic granulomas and hemorrhage'),
# 0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),  3: ('df', 'dermatofibroma')}
#
# # Load the image
# img = cv2.imread("/content/HAM10000_images_part_1/ISIC_0024333.jpg")
#
# # Display the image
# cv2_imshow(img)
#
# # Predicting for this:
# img = cv2.resize(img, (28, 28))
# result = model.predict(img.reshape(1, 28, 28, 3))
# max_prob = max(result[0])
# class_ind = list(result[0]).index(max_prob)
# class_name = classes[class_ind]
# print(class_name)

import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pickle
from keras.src.applications.vgg19 import VGG19
from keras.applications.imagenet_utils import preprocess_input

vgg16 = VGG19(include_top=False, weights='imagenet')

app = FastAPI()

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

model = pickle.load(open('skin_model.pkl', 'rb'))
class_names = ['Acne and Rosacea Photos',
               'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
               'Atopic Dermatitis Photos',
               'Eczema Photos',
               'Nail Fungus and other Nail Disease',
               'Psoriasis pictures Lichen Planus and related diseases']


def load_img(img_path):
    images = []
    img = cv2.imread(img_path)
    img = cv2.resize(img, (100, 100))
    images.append(img)
    x_test = np.asarray(images)
    test_img = preprocess_input(x_test)
    features_test = vgg16.predict(test_img)
    num_test = x_test.shape[0]
    f_img = features_test.reshape(num_test, 4608)

    return f_img


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
        img_path = "/upload/" + filename
        img = load_img(img_path)
        predicted_index = np.argmax(model.predict(img))
        predicted_class_name = class_names[predicted_index]
        print("Image Uploaded Successfully!")
        # Delete the uploaded image file
        os.remove(filename)
        return JSONResponse(content={"message": "success", "prediction": predicted_class_name})
    except Exception as e:
        # If an error occurs, return error message
        return JSONResponse(content={"message": "error", "error": str(e)})

# To start the server write: uvicorn main:app --reload
