import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import inception_v3

def get_feature_extractor(layer_settings:dict=None):

    if layer_settings is None:
        layer_settings = {
            "mixed4": 0.0,
            "mixed5": 1.5,
            "mixed6": 2.0,
            "mixed7": 0.5,
        }

    # Build an InceptionV3 model loaded with pre-trained ImageNet weights
    model = inception_v3.InceptionV3(weights="imagenet", include_top=False)

    # Get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict(
        [
            (layer.name, layer.output)
            for layer in [model.get_layer(name) for name in layer_settings.keys()]
        ]
    )

    # Set up a model that returns the activation values for every target layer
    # (as a dict)
    feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

    return feature_extractor
