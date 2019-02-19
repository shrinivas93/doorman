import base64
import datetime
import os

import flask
import jsons
import tensorflow as tf
from flask import jsonify

from Camera import Camera
from YoloObjectDetector import YoloObjectDetector

graph = tf.get_default_graph()

camera_captures_base_path = os.path.abspath('/doorman/camera_captures')
detection_output_base_path = os.path.abspath('/doorman/detection_output')

if not os.path.exists(camera_captures_base_path):
    os.makedirs(camera_captures_base_path)

if not os.path.exists(detection_output_base_path):
    os.makedirs(detection_output_base_path)

execution_path = os.getcwd()
print('Running application at ', execution_path)

print('Initializing Camera...')
camera = Camera(camera_index=0)
print('Camera initialized')

print('Loading Yolov3 object detection model...')
# Download this model from https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5
object_detector = YoloObjectDetector(
    yolo_model_path=os.path.join(execution_path, 'yolo.h5'))
print('Yolov3 model loaded')

print('Starting API server...')
app = flask.Flask(__name__)


@app.route('/detect', methods=['GET'])
def detect():
    current_timestamp = '{0:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())
    picture_file_path = 'capture_' + current_timestamp + '.jpg'
    status = camera.capture(os.path.join(
        camera_captures_base_path, picture_file_path))
    detections = []
    detected_objects_location = []
    response_data = {
        "detections": [],
        "cameraCapture": "",
        "detectedObjects": []
    }
    if status:
        with graph.as_default():
            print('A: ', os.path.join(camera_captures_base_path, picture_file_path))
            print('B: ', os.path.join(
                detection_output_base_path, picture_file_path))
            detections, detected_objects_location = object_detector.detect(os.path.join(camera_captures_base_path, picture_file_path), os.path.join(
                detection_output_base_path, picture_file_path))

    response_data["cameraCapture"] = 'data:image/jpeg;base64,{}'.format(
        base64.b64encode(open(os.path.join(camera_captures_base_path, picture_file_path), "rb").read()).decode('utf-8'))

    for detection in detections:
        detection_data = {}
        detection_data["type"] = detection["name"]
        detection_data["score"] = float(detection["percentage_probability"])
        detection_data["boundingBox"] = list(
            map(int, list(detection["box_points"])))
        response_data["detections"].append(detection_data)

    for detected_object in detected_objects_location:
        response_data["detectedObjects"].append('data:image/jpeg;base64,{}'.format(
            base64.b64encode(open(detected_object, "rb").read()).decode('utf-8')))

    return jsonify(response_data)


print('API server started')
app.run(threaded=True)
