import numpy as np
import scipy.misc
import cv2


def get_images(filename, is_crop, fine_size, images_norm):
    img = cv2.imread(filename)

    if images_norm:
        img = (img-127.5)/127.5

    images_ = []

    if is_crop:
        size = img.shape
        h_ = int(size[0]) // fine_size
        w_ = int(size[1]) // fine_size

        for h in range(h_):
            for w in range(w_):
                image_temp = img[h*fine_size:(h+1)*fine_size, w*fine_size:(w+1)*fine_size, :]
                images_.append(image_temp)

    else:
        img = np.array(img).astype(np.float32)
        images_.append(img)

    return images_


def save_images(images, size, filename, images_norm):
    h_, w_ = size[0], size[1]
    img_out = None
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


def blur_images(image, images_norm, output_size):
    input_ = cv2.GaussianBlur(image, (7, 7), 0.9)
    image_ = image
    if images_norm:
        input_ = (input_-127.5)/127.5
        image_ = (image-127.5)/127.5
    padding = int(input_.shape[0] - output_size)//2
    image_ = image_[padding:padding+output_size, padding:padding+output_size, :]
    return input_, image_


def get_sample_image(filename, input_size, output_size, images_norm):
    assert input_size >= output_size
    image = cv2.imread(filename)
    size = image.shape
    stride = output_size
    h = (size[0] - input_size + 1) // stride + 1
    w = (size[1] - input_size + 1) // stride + 1
    padding = int(input_size - output_size)//2
    # input_, sample_ = blur_images(img, images_norm, output_size)

    input_ = cv2.GaussianBlur(image, (7, 7), 0.9)
    sample_ = image
    if images_norm:
        input_ = (input_-127.5)/127.5
        sample_ = (image-127.5)/127.5

    inputs_ = []
    samples_ = []
    for x in range(0, size[0] - input_size + 1, stride):
        for y in range(0, size[1] - input_size + 1, stride):
            inputs_.append(input_[x:x + input_size, y:y + input_size])
            samples_.append(sample_[x+padding:x+padding + output_size, y+padding:y+padding + output_size])

    inputs_ = np.array(inputs_).astype(np.float32)
    samples_ = np.array(samples_).astype(np.float32)

    return h, w, inputs_, samples_
