# Deep Dream - Dreaming with a Neural Network in Keras

introduced by Alexander Mordvintsev from Google in July 2015, **Deep Dream** is a technique for visualizing the patterns learned by a neural network. 

It does so by maximizing the activations of specific layer(s) for a given input image using gradient descent.

Original image             |  Image with patterns
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/parmarsuraj99/deep-dream-keras/master/inputs/sky.jpg)  |  ![](https://raw.githubusercontent.com/parmarsuraj99/deep-dream-keras/master/results/sky_dream.jpg)

The idea is to maximize loss such that the image increasingly activates the selected layers. Shallow layers focus more on low-level patterns like lines and abstract geometrical patterns. Deeper layers, meanwhile, focus more on patterns specific to the images the model  was trained on. We can use this fact to select the layers and get different outputs.

## Overview

In this tutorial we will use a pre-trained Convolutional Neural Network model called "[InceptionV3](https://keras.io/api/applications/inceptionv3/)" from tf.keras .

We'll use **Spell Runs** to run a script from the code repository that takes an image and saves an image after applying the deep dream algorithm.

Code for the tutorial [parmarsuraj99/deep-dream-keras](https://github.com/parmarsuraj99/deep-dream-keras). This repo also contains some sample input images that we can use for initial explorations.

## Let's dream

```sh
$ spell run --machine-type CPU \
    --pip matplotlib \
    --pip numpy \
    --pip opencv-python \
    --pip tensorflow_gpu==2.3.0 \
    --pip tqdm \
    'python src/dream.py \
        --src_img inputs/sky.jpg \
        --result_img results/sky_dream.jpg'
```

Run logs are outputted directly to the CLI like so:

```sh
model loaded
Dreaming
('results', 'aus_dream.jpg')
(457, 685)
(326, 489)
Processing octave 0 with shape (326, 489)
... Loss value at step 1: 0.41
... Loss value at step 2: 0.62
    ...
    ...
... Loss value at step 8: 3.54
... Loss value at step 9: 3.88
Total time: 46.362966537475586 s
✨ Run is saving
Scanning for modified or new files from the run
Saving '/spell/results/aus_dream.jpg
✨ Run is pushing
Saving build environment for future runs
✨ Total run time: 1m25.039473s
✨ Run 41 complete

```

Outputs from the runs are saved in **`Resources > Runs > #run`**.

## Uploading and mounting an image

This run was performed using a default image, but you might want to experiment with your input image. You can upload an image to  Spell and mount it into the run

```sh
$ spell upload -n deep_dream inputs
```

![Image upload screenshot](https://raw.githubusercontent.com/parmarsuraj99/deep-dream-keras/master/spell_images/uploads.png)

```sh
$ python src/dream.py \
    --src_img /spell/flowers_pixel.jpg \
    --result_img results/flower_dream.jpg
```

Here's an alternative result based on a picture I uploaded to Spell:

![Alternative result](https://raw.githubusercontent.com/parmarsuraj99/deep-dream-keras/master/results/flowers_pixel_dream.jpg)
