import os
from datetime import datetime as dt
from pdb import set_trace as bp

from PIL import Image
import h5py
import numpy as np
import matplotlib.pyplot as plt

IMAGE_SIZE = 256


def read_one_image(file_name):
    im_frame = Image.open(file_name)
    frame = np.array(im_frame.getdata(), dtype=int)
    frame = frame.reshape(im_frame.height, im_frame.width, 3)
    return frame


def multi_crop(
        img,
        stride=128,
):
    height, width, _ = img.shape
    cropped_images = []
    for h in range((height - IMAGE_SIZE)//stride + 1):
        for w in range((width - IMAGE_SIZE)//stride + 1):
            _cropped = img[
                h * stride: h * stride + IMAGE_SIZE,
                w * stride: w * stride + IMAGE_SIZE,
                :]
            cropped_images.append(_cropped)

    if (height - IMAGE_SIZE) % stride > stride / 2:
        for w in range((width - IMAGE_SIZE)//stride + 1):
            _cropped = img[
                height - IMAGE_SIZE: height,
                w * stride: w * stride + IMAGE_SIZE,
                :]
            cropped_images.append(_cropped)

    if (width - IMAGE_SIZE) % stride > stride / 2:
        for h in range((height - IMAGE_SIZE)//stride + 1):
            _cropped = img[
                h * stride: h * stride + IMAGE_SIZE,
                width - IMAGE_SIZE: width,
                :]
            cropped_images.append(_cropped)

    return cropped_images


def save_images(
        data,
        file_name='data/flowers.hdf5',
        category='flower',
        mode='a',
):
    with h5py.File(file_name, mode) as f:
        f.create_dataset(category, data=data)


def load_images(file_name='data/flowers.hdf5'):
    data = h5py.File(file_name, 'r')
    return data


def read_multiple_image():
    directory = 'data/flowers/'
    images = {}
    for category in ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']:
        images[category] = []
        path = directory + category
        for image_name in os.listdir(path):
            if '.png' not in image_name and '.jpg' not in image_name:
                continue

            img = read_one_image(path + '/' + image_name)
            if img.shape[0] < IMAGE_SIZE or img.shape[1] < IMAGE_SIZE:
                continue

            cropped_images = multi_crop(img)
            images[category] += cropped_images
            for _cropped in cropped_images:
                if _cropped.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
                    plt.imshow(img)
                    print(image_name, _cropped.shape, img.shape)
    return images


def batch_process_image():
    t1 = dt.now()
    images = read_multiple_image()
    t2 = dt.now()
    print("Read image takes:", (t2 - t1).total_seconds() / 3600)

    for category, _images in images.items():
        save_images(
            data=_images,
            category=category,
        )
    t3 = dt.now()
    print("Save image takes:", (t3 - t2).total_seconds() / 3600)


def load_data():
    images = load_images()
    for category, data in images.items():
        pass
    return images


def test():
    batch_process_image()
    images = load_images()
    return images


def test_one():
    file_name = 'data/flowers/daisy/3386988684_bc5a66005e.jpg'
    img = read_one_image(file_name)
    cropped_images = multi_crop(img)
    
    save_images(cropped_images)
    new_images = load_images()

    return cropped_images, new_images


if __name__ == "__main__":
    batch_process_image()
    # plt.imshow(img)
