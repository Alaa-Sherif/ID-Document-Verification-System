from fastapi import FastAPI, File, UploadFile, Response
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from PIL import Image
import io
import base64
import numpy as np
import cv2
from tensorflow import keras
from keras.models import load_model
from code_files.pipelines import id_pipeline, bachelor_cert_pipeline, image_pipeline, military_cert_pipeline, CustomError
# from code_files.pipelines import image_pipeline, CustomError, id_pipeline
from fastapi.responses import StreamingResponse


mc_classifier = load_model('code_files/military_certificate/classification/CNN_MC_Classifier.h5')
id_classifier = load_model('code_files/id/classification/CNN_ID_Classifier.h5')
bc_Classifier = load_model("code_files/graduation_certificate/classification/CNN_BC_Classifier2.h5")

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

def success_response(new_image, response):
    _, img_bytes = cv2.imencode(".jpg", new_image)
    image_encode = base64.b64encode(img_bytes.tobytes())
    response['is_valid'] = "1"
    response['image_encoding'] = image_encode
    return response

def image_file_success_response(new_image, response):
        response['is_valid'] = "1"
        _, img_bytes = cv2.imencode(".jpg", new_image)
        img_response = Response(content=img_bytes.tobytes(), media_type="image/jpg", headers=response)
        return img_response

def failure_response(e, response):
    error_msg = e.message
    response['is_valid'] = "0"
    response['error'] = error_msg
    return response

@app.post("/validate")
async def validate_document(doc_id: str, doc_type: str, image: UploadFile = File(...)):
    extension = image.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return {'error': 'Wrong_Format'}

    contents = await image.read()

    # img = Image.open(io.BytesIO(contents))
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    response = {}
    response['id'] = doc_id
    response['type'] = doc_type

    match doc_type:
        case '0':
            try:
                new_image = image_pipeline(img, doc_id)
                # response = success_response(new_image, response)
                response = image_file_success_response(new_image, response)

            except CustomError as e:
                response = failure_response(e, response)
            
        case '1': 
            try:
                name, id, new_image = id_pipeline(img, id_classifier)
                response['name'] = str(name.encode('utf-8'))
                response['national_id'] = str(id)
                # response = success_response(new_image, response)
                response = image_file_success_response(new_image, response)

            except CustomError as e:
                response = failure_response(e, response)

        case '2':
            try:
                new_image = bachelor_cert_pipeline(img, bc_Classifier)
                # response = success_response(new_image, response)
                response = image_file_success_response(new_image, response)

            except CustomError as e:
                response = failure_response(e, response)

        case '3':
            try:
                military_result, new_image = military_cert_pipeline(img, mc_classifier)
                response['military_status'] = military_result
                # response = success_response(new_image, response)
                response = image_file_success_response(new_image, response)

            except CustomError as e:
                response = failure_response(e, response)

        # case 'Personal_Image':
        #     pass
        # case 'Personal_Id':
        #     pass
        # case 'Bachelor_Certificate':
        #     pass
        # case 'Military_Certificate':
        #     pass


    return response
