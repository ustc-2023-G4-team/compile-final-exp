import numpy as np
import cv2
import os
import time
from multiprocessing import Pool
import multiprocessing
from PyRuntime import OMExecutionSession

def init_model():
    global session
    model = './lenet.so'
    time_start = time.time()
    session = OMExecutionSession(model, use_default_entry_point=True)
    time_end = time.time()
    print('time cost: {}'.format(time_end - time_start))

def run_inference(filename):
    picture_path = os.path.join(image_folder, filename)
    img = cv2.imread(picture_path)
    input = np.array(img[:,:,0], np.dtype(np.float32))
    outputs = session.run(input)
    output_tensor = outputs[0]
    output_array = np.squeeze(output_tensor)
    predict_label = np.argmax(output_array)
    # print('predict label: {}'.format(predict_label))

if __name__ == '__main__':
    image_folder = './Data/mnist_images'
    filenames = os.listdir(image_folder)

    num_processes = multiprocessing.cpu_count()
    pool = Pool(processes=num_processes, initializer=init_model)

    time_start = time.time()
    pool.map(run_inference, filenames)
    pool.close()
    pool.join()
    time_end = time.time()

    print('time cost: {}'.format(time_end - time_start))
