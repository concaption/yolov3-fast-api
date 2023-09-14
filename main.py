#!/usr/bin/env python
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from enum import Enum

import numpy as np

import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

from io import BytesIO


app = FastAPI(
    title="Fast API for YOLOv3-tiny",
    description="Fast API for YOLOv3-tiny",
    version="1.0.0",
)


class Model(str, Enum):
    yolov3tiny = "yolov3tiny"
    yolov3 = "yolov3"


async def predict_yolov3(file, model_type="yolov3"):
    """
    Function to predict image class.
    """
    file_name = file.filename

    # read image as a stream of bytes
    image_stream = BytesIO(file.file.read())

    # start the stream from the begining (position zero)
    image_stream.seek(0)

    #  write the stream data to a numpy array
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)

    # decode the numpy array as an image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # apply object detection
    bbox, label, conf = cv.detect_common_objects(image, model=model_type)

    # create image that includes bounding boxes and labels
    output_image = draw_bbox(image, bbox, label, conf)

    # save output image
    cv2.imwrite(f"outputs/{file_name}", output_image)

    file_image = open(f"outputs/{file_name}", mode="rb")

    return StreamingResponse(file_image, media_type="image/jpeg")


predict_yolov3tiny = lambda file: predict_yolov3(file, model_type="yolov3tiny")


@app.get("/")
async def root():
    """
    Root endpoint.
    """
    return {
        "message": "Congratulations! Your API is working as expected. Now head over to http://localhost:8000/docs."
    }


@app.post("/predict")
async def predict(model: Model, file: UploadFile = File(...)):
    """
    Endpoint to predict an image.
    """
    file_name = file.filename
    file_extension = file_name.split(".")[-1] in ("jpg", "jpeg", "png")
    if not file_extension:
        raise HTTPException(
            status_code=415,
            detail="Invalid image type. Allowed image types are 'jpg', 'jpeg' and 'png'.",
        )

    if model == Model.yolov3tiny:
        return await predict_yolov3tiny(file)
    elif model == Model.yolov3:
        return await predict_yolov3(file)
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid model type. Model type can be yolov3tiny or yolov3.",
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
