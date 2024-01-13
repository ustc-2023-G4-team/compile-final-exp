import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def transfer_images():
    # 读取图像文件
    filename = './t10k-images.idx3-ubyte'
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)

    # 将图像数据转换为二维数组
    images = data.reshape(-1, 28, 28)

    # 创建保存图像的目录
    save_dir = 'mnist_images'
    os.makedirs(save_dir, exist_ok=True)

    # 保存图像
    for i in range(len(images)):
        image = images[i]
        save_path = os.path.join(save_dir, f'image_{i}.png')
        plt.imsave(save_path, image, cmap='gray')

def transfer_labels():
    # 读取标签文件
    filename = './t10k-labels.idx1-ubyte'
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    # 将标签数据保存为文本文件
    save_path = 'labels.txt'
    with open(save_path, 'w') as f:
        for label in data:
            f.write(str(label) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', help='transfer images', default=False)
    parser.add_argument('--labels', help='transfer labels', default=False)
    args = parser.parse_args()
    images = args.images
    labels = args.labels
    if images:
        transfer_images()
    if labels:
        transfer_labels()

