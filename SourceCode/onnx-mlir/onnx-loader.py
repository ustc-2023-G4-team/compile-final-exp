import numpy as np
import cv2
import time
import os
import onnxruntime as ort

time_start = time.time()
model_path = './lenet.onnx'
session = ort.InferenceSession(model_path)

image_folder = './Data/mnist_images'
for filename in os.listdir(image_folder):
    picture_path = os.path.join(image_folder, filename)
    image = cv2.imread(picture_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.expand_dims(image, axis=-1)

    input_data = np.array(image, dtype=np.float32)
    input_data = np.transpose(input_data, [2, 0, 1])
    input_data = np.expand_dims(input_data, axis=0)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    result = session.run([output_name], {input_name: input_data})

    output_tensor = result[0]
    output_array = np.squeeze(output_tensor)
    predict_label = np.argmax(output_array)
    print('predict label: {}'.format(predict_label))
    
time_end = time.time()
print('time cost: {}'.format(time_end - time_start))