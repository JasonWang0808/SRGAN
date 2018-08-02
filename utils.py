import numpy as np
import scipy.misc
import cv2


def get_images(filename, is_crop, fine_size, images_norm):
    img = cv2.imread(filename)
    if is_crop:
        size = img.shape
        start_h = int((size[0] - fine_size)/2)
        start_w = int((size[1] - fine_size)/2)
        img = img[start_h:start_h+fine_size, start_w:start_w+fine_size,:]
    img = np.array(img).astype(np.float32)
    if images_norm:
        img = (img-127.5)/127.5
    return img


def save_images(images, size, filename, images_norm):
    h_, w_ = size[0], size[1]
    img_out = None
    # img_samples = np.zeros([h_*self.config.fine_size, w_*self.config.fine_size, 3])
    for h in range(h_):
        img_samples = images[h * w_]
        for w in range(1, w_):
            img_samples = np.concatenate((img_samples, images[h * w_ + w]), axis=1)
        if img_out is None:
            img_out = img_samples
        else:
            img_out = np.concatenate((img_out, img_samples), axis=0)
    if images_norm is True:
        img_out = img_out * 127.5 + 127.5
    return cv2.imwrite(filename, img_out)


def blur_images(image, images_norm):
    input_ = cv2.GaussianBlur(image, (7, 7), 0.9)
    image_ = image
    if images_norm:
        input_ = (input_-127.5)/127.5
        image_ = (image-127.5)/127.5
    return input_, image_


def get_sample_image(filename, fine_size, images_norm):
    img = cv2.imread(filename)
    size = img.shape
    h = size[0] // fine_size
    w = size[1] // fine_size
    input_, sample_ = blur_images(img, images_norm)
    inputs_ = []
    samples_ = []
    for x in range(0, size[0] - fine_size + 1, fine_size):
        for y in range(0, size[1] - fine_size + 1, fine_size):
            inputs_.append(input_[x:x + fine_size, y:y + fine_size])
            samples_.append(sample_[x:x + fine_size, y:y + fine_size])

    inputs_ = np.array(inputs_).astype(np.float32)
    samples_ = np.array(samples_).astype(np.float32)

    return h, w, inputs_, samples_