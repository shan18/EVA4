import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

from PIL import Image
from skimage.transform import resize
from keras.models import load_model
from tqdm.autonotebook import tqdm

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


def predict(model, images, minDepth=10, maxDepth=1000, batch_size=100):
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


def save_output(outputs, path, name_list):
    for idx, output in enumerate(outputs):
        output_img = Image.fromarray(
            (output[:, :, 0] * 255).astype(np.uint8)
        )
        output_img.save(os.path.join(path, f'{name_list[idx]}.jpeg'))


def predict_batch(model, images_list, input_path, output_path, batch_size):
    # Load images
    inputs = load_images([
        os.path.join(input_path, x)
        for x in images_list
    ])

    # Compute Results
    outputs = predict(model, inputs, batch_size=batch_size)

    # Save Results
    save_output(outputs, output_path, [
        os.path.splitext(x)[0]
        for x in images_list
    ])


def load_densenet(model_path):
    # Custom object needed for inference and training
    custom_objects = {
        'BilinearUpSampling2D': BilinearUpSampling2D,
        'depth_loss_function': None
    }

    # Load model into GPU / CPU
    print('Loading model...')
    model = load_model(model_path, custom_objects=custom_objects, compile=False)
    print(f'\nModel loaded ({model_path}).')

    return model


def depth_map(model, input_path, batch_size=100):
    output_path = os.path.join(
        os.path.dirname(input_path), os.path.basename(input_path) + '_depth_map'
    )
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Get list of input images
    # This will also discard the images which have already been predicted and
    # are present in output_path
    images_list = list(
        set(os.listdir(input_path)) - set(os.listdir(output_path))
    )

    print('\nPredicting depth maps...')
    if len(images_list) > 0:
        if len(images_list) > batch_size:
            for batch_idx in tqdm(range(0, len(images_list), batch_size)):
                predict_batch(
                    model, images_list[batch_idx:batch_idx + batch_size],
                    input_path, output_path, batch_size
                )
        else:
            predict_batch(model, images_list, input_path, output_path, len(images_list))
    
        print(f'Predictions saved in {output_path}')
    else:
        print(f'All the images have already been predicted and saved in {output_path}')
