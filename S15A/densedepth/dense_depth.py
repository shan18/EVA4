import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

from PIL import Image
from skimage.transform import resize
from keras.models import load_model

from densedepth.layers import BilinearUpSampling2D


def load_images(images_list):
    loaded_images = []
    for file in images_list:
        x = np.clip(
            np.asarray(
                Image.open(file),
                dtype=float
            ) / 255, 0, 1)
        loaded_images.append(x)
    return np.stack(loaded_images, axis=0)


def predict(model, images, minDepth=10, maxDepth=1000, batch_size=2):
    # Support multiple RGBs, one RGB image, even grayscale 
    if len(images.shape) < 3:
        images = np.stack((images, images, images), axis=2)
    if len(images.shape) < 4:
        images = images.reshape(
            (1, images.shape[0], images.shape[1], images.shape[2])
        )
    
    # Compute predictions
    predictions = model.predict(images, batch_size=batch_size)

    # Put in expected range
    return np.clip(
        maxDepth / predictions, minDepth, maxDepth
    ) / maxDepth


def to_multichannel(i):
    if i.shape[2] == 3:
        return i
    i = i[:, :, 0]
    return np.stack((i, i, i), axis=2)


def save_output(outputs, path, name_list):
    if not os.path.exists(path):
        os.makedirs(path)
    for idx, output in enumerate(outputs):
        output_img = Image.fromarray(
            (to_multichannel(output) * 255).astype(np.uint8)
        )
        output_img.save(os.path.join(path, f'{name_list[idx]}.jpeg'))


def depth_map(model_path, input_path):
    # Custom object needed for inference and training
    custom_objects = {
        'BilinearUpSampling2D': BilinearUpSampling2D,
        'depth_loss_function': None
    }

    print('Loading model...')

    # Load model into GPU / CPU
    model = load_model(model_path, custom_objects=custom_objects, compile=False)
    print(f'\nModel loaded ({model_path}).')

    # Input images
    images_list = os.listdir(input_path)
    inputs = load_images([os.path.join(input_path, x) for x in images_list])
    print(f'\nLoaded ({inputs.shape[0]}) images of size {inputs.shape[1:]}.')

    # Compute results
    print('\nPredicting depth maps...')
    outputs = predict(model, inputs)

    # Save Results
    output_path = os.path.join(
        os.path.dirname(input_path), os.path.basename(input_path) + '_depth_mask'
    )
    save_output(outputs, output_path, [
        os.path.splitext(x)[0] for x in images_list
    ])
    print(f'Predictions saved in {output_path}')
