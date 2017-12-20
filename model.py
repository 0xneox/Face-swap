from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

from pixel_shuffler import PixelShuffler

optimizer = Adam( lr=5e-5, beta_1=0.5, beta_2=0.999 )

IMAGE_SHAPE = (64,64,3)
ENCODER_DIM = 1024

def conv( filters ):
    def block(x):
        x = Conv2D( filters, kernel_size=5, strides=2, padding='same' )(x)
        x = LeakyReLU(0.1)(x)
        return x
    return block

def upscale( filters ):
    def block(x):
        x = Conv2D( filters*4, kernel_size=3, padding='same' )(x)
        x = LeakyReLU(0.1)(x)
        x = PixelShuffler()(x)
        return x
    return block
