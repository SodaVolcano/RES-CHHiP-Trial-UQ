"""
Custom layers for the neural network models.
"""

from uncertainty.data.preprocessing import crop_nd

from tensorflow import keras  # type: ignore
import tensorflow as tf


class CentreCrop3D(keras.layers.Layer):
    """
    Crop the input tensor to the new shape
    """

    def __init__(self, new_shape: tf.TensorShape) -> None:
        """new_shape is the shape of a single instance"""
        super(CentreCrop3D, self).__init__()
        self.out_shape = new_shape

    def call(self, inputs):
        new_shape = (inputs.shape[0],) + self.out_shape
        return crop_nd(inputs, new_shape)


class MCDropout(keras.layers.Dropout):
    def call(self, inputs, training: bool = False):
        # Always use dropout even in testing
        return super().call(inputs, training=True)
