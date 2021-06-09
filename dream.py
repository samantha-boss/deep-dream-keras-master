import os, time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import inception_v3

import matplotlib.pyplot as plt
from utils import preprocess_image, deprocess_image
from model import get_feature_extractor

import argparse

print("Libraries Loaded!")
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# You can tweak these setting to obtain new visual effects.
layer_settings = {
    "mixed4": 0.0,
    "mixed5": 1.5,
    "mixed6": 2.0,
    "mixed7": 0.5,
}

# Playing with these hyperparameters will also allow you to achieve new effects
step = 0.01  # Gradient ascent step size
num_octave = 3  # Number of scales at which to run gradient ascent
octave_scale = 1.4  # Size ratio between scales
iterations = 10  # Number of ascent steps per scale
max_loss = 15.0

#----------------------------------------------------------------------

def compute_loss(input_image, feature_extractor):
    features = feature_extractor(input_image)
    # Initialize the loss
    loss = tf.zeros(shape=())
    for name in features.keys():
        coeff = layer_settings[name]
        activation = features[name]
        # We avoid border artifacts by only involving non-border pixels in the loss.
        scaling = tf.reduce_prod(tf.cast(tf.shape(activation), "float32"))
        loss += coeff * tf.reduce_sum(tf.square(activation[:, 2:-2, 2:-2, :])) / scaling
    return loss


@tf.function
def gradient_ascent_step(img, feature_extractor, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, feature_extractor)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads /= tf.maximum(tf.reduce_mean(tf.abs(grads)), 1e-6)
    img += learning_rate * grads
    return loss, img


def gradient_ascent_loop(img, feature_extractor, iterations, learning_rate, max_loss=max_loss):
    for i in range(iterations):
        loss, img = gradient_ascent_step(img, feature_extractor, learning_rate)
        if max_loss is not None and loss > max_loss:
            break
        print("... Loss value at step %d: %.2f" % (i, loss))
    return img

#----------------------------------------------------------------------

def dream_on(original_img, feature_extractor, output_name="result.jpg"):
    #original_img = preprocess_image(base_image_path)
    original_shape = original_img.shape[1:3]

    successive_shapes = [original_shape]
    for i in range(1, num_octave):
        shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
        print(shape)
        successive_shapes.append(shape)
    successive_shapes = successive_shapes[::-1]
    shrunk_original_img = tf.image.resize(original_img, successive_shapes[0])

    img = tf.identity(original_img)  # Make a copy
    for i, shape in enumerate(successive_shapes):
        print("Processing octave %d with shape %s" % (i, shape))
        img = tf.image.resize(img, shape)
        img = gradient_ascent_loop(
            img, feature_extractor=feature_extractor,
             iterations=iterations, learning_rate=step, max_loss=max_loss
        )
        upscaled_shrunk_original_img = tf.image.resize(shrunk_original_img, shape)
        same_size_original = tf.image.resize(original_img, shape)
        lost_detail = same_size_original - upscaled_shrunk_original_img

        img += lost_detail
        shrunk_original_img = tf.image.resize(original_img, shape)

    keras.preprocessing.image.save_img(output_name, deprocess_image(img.numpy()))

#----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Deep Dream tutorial")
    parser.add_argument("--src_img", default="sky.jpg", required=True, type=str, help="Source image to perform deep dram on")
    parser.add_argument("--result_img", default="results/dream_result.jpg", type=str, help="Result image to perform deep dram on")
    parser.add_argument("--downscale_factor", default=1, type=float, help="Downscale Factor")


    args = parser.parse_args()

    proc = preprocess_image(args.src_img)

    if args.downscale_factor > 1:
        print(proc.shape)
        new_shape = [int(proc.shape[1]//args.downscale_factor), 
                    int(proc.shape[2]//args.downscale_factor)]
        proc = tf.image.resize(proc, new_shape)


    print(proc.shape)

    model = get_feature_extractor(layer_settings)
    print("model loaded\nDreaming")

    print(os.path.split(args.result_img))

    if not os.path.isdir(args.result_img.split("/")[0]):
        try:
            d_dir = args.result_img.split("/")[0]
            os.mkdir(d_dir)
            print(f"created directory: {d_dir}")
        except:
            print("couldn't create directory")

    dream_on(proc, model, args.result_img)


if __name__ == "__main__":
    st = time.time()
    main()
    print(f"Total time: {time.time()-st} s")
