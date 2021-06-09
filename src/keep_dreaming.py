import os, time
import cv2
import numpy as np
from PIL import Image
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
    "mixed3": 0.5,
    "mixed4": 1.8,
    "mixed5": 1.5,
    "mixed6": 1.5,
    "mixed7": 1.9,
}

# Playing with these hyperparameters will also allow you to achieve new effects
step_size = 0.01  # Gradient ascent step size
num_octave = 3  # Number of scales at which to run gradient ascent
octave_scale = 1.4  # Size ratio between scales
optim_steps = 10  # Number of ascent steps per scale
max_loss = 1500.0

#----------------------------------------------------------------------

def loss_fn(image, model):
    features = model(image)
    
    #print(features)    
    loss = tf.zeros(shape=())
    for layer in features.keys():
        coeff = layer_settings[layer]
        activation = features[layer]
        scaling = tf.reduce_prod(tf.cast(tf.shape(activation), tf.float32))
        loss+=coeff*tf.reduce_sum(tf.square(activation[:, 2:-2, 2:-2, :]))/scaling
    
    return loss


@tf.function
def _gradient_ascent(img, model:tf.keras.Model, step_size):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = loss_fn(img, model)
    grads = tape.gradient(loss, img)
    
    #Normalizing  gradients: Crucial
    grads /= tf.maximum(tf.math.reduce_mean(tf.abs(grads)), 1e-8)
    img += step_size * grads
    img = tf.clip_by_value(img, -1, 1)
    
    return loss, img

def gradient_ascent_loop(img, model, optim_steps, step_size, max_loss=None):
    for i in range(optim_steps):
        loss, img = _gradient_ascent(img, model, step_size)
        
        if max_loss and loss>max_loss:
            print("max loss reached")
            break
        
        print(f"loss value at step {i}: {loss}")
    return img

#----------------------------------------------------------------------

def dream_on(original_img, feature_extractor, output_dir, iterations=1000, save_every=10, downscale_factor=2):

    #processed_img = preprocess_image(original_img)
    processed_img = original_img
    processed_img = tf.image.resize(processed_img, 
            (int(processed_img.shape[1]/downscale_factor), int(processed_img.shape[2]/downscale_factor))
        )
    img =  processed_img

    x_size, y_size = int(processed_img.shape[1]), int(processed_img.shape[2])
    print(f"x_size: {x_size}, y_size:{y_size}")

    for i in range(iterations):
        

        files = os.listdir(f"{output_dir}")
        files = sorted(files, key=lambda x: int(x.split("_")[3].split(".")[0]))
        print(f"recent saves: {files[-2:]}")
    
        if os.path.isfile(f"{output_dir}/dream_{img.shape[1]}_{img.shape[2]}_{i}" + ".jpg"):
            print(f"{output_dir}/dream_{img.shape[1]}_{img.shape[2]}_{i}" + ".jpg Exist")

        elif len(os.listdir(f"{output_dir}"))==0:
            img = processed_img
            #img = tf.keras.preprocessing.image.img_to_array(img)
            tf.keras.preprocessing.image.save_img(f"{output_dir}/dream_{img.shape[1]}_{img.shape[2]}_{i}" + ".jpg", deprocess_image(img.numpy()))
        else:
            lastfile = files[-1]
        
            img = tf.keras.preprocessing.image.load_img(f"{output_dir}/{lastfile}")
            img = tf.keras.preprocessing.image.img_to_array(img)
            
            x_trim = 2
            y_trim = 2

            print(img.shape)
            #img = img[0:x_size-x_trim, 0:y_size-y_trim]
            img = tf.image.central_crop(img, central_fraction=0.99)
            img = tf.image.resize(img, (x_size, y_size))
            print(img.shape)

            #kernel = np.ones((5,5),np.float32)/25
            #img = cv2.filter2D(np.array(img),-1,kernel)
            #img = cv2.GaussianBlur(np.array(img), (9, 9), 0)
            #img = cv2.resize(img, (y_size, x_size))

            print(img.shape)
            img = tf.expand_dims(img, axis=0)
            img = inception_v3.preprocess_input(img)
            print(i%save_every)

            img = gradient_ascent_loop(img, feature_extractor, optim_steps, step_size, max_loss=None)

            if save_every>0 and i%save_every==0:
                deproc_img = deprocess_image(img.numpy())

                deproc_img = cv2.GaussianBlur(deproc_img, (3, 3), 0)

                tf.keras.preprocessing.image.save_img(f"{output_dir}/dream_{img.shape[1]}_{img.shape[2]}_{i}" + ".jpg", deproc_img)
                print(f"-------dream_{img.shape[1]}_{img.shape[2]}_{i}" + ".jpg-------")

#----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Deep Dream tutorial")
    parser.add_argument("--src_img", default="sky.jpg", required=True, type=str, help="Source image to perform deep dram on")
    parser.add_argument("--directory", default="../dream_dir", type=str, help="Result directory to save intermediate images")
    parser.add_argument("--iterations", default="1000", type=int, help="How long to dream.")
    parser.add_argument("--save_every", default="1", type=int, help="Saving image after every _ iterations")
    parser.add_argument("--downscale_factor", default="3", type=int, help="Downscale factor for reducing image scale")
    parser.add_argument("--overwrite_save_dir", default=False, type=bool, help="Delete all files in selected directory")

    args = parser.parse_args()

    proc = preprocess_image(args.src_img)
    print(proc.shape)

    model = get_feature_extractor(layer_settings)
    print("model loaded\nDreaming")

    if not os.path.isdir(args.directory):
        try:
            os.mkdir(args.directory)
            print(f"created directory \"{args.directory}\"")
        except:
            print("couldn't create directory")

    if  len(os.listdir(args.directory))>0 and args.overwrite_save_dir==True:
        for f in os.listdir(args.directory):
            os.remove(os.path.join(args.directory, f))
        print("Directory cleaned")

    dream_on(proc, model, args.directory, iterations=args.iterations, save_every=args.save_every, downscale_factor=args.downscale_factor)


if __name__ == "__main__":
    st = time.time()
    main()
    print(f"Total time: {time.time()-st} s")

