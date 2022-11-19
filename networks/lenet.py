from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K


class LeNet:
    @staticmethod
    def build(num_channels, img_rows, img_cols, num_classes,
              activation="relu", weights_path=None):
        # initialize the model
        model = Sequential()
        input_shape = (img_rows, img_cols, num_channels)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            input_shape = (num_channels, img_rows, img_cols)

        # define the first set of CONV => ACTIVATION => POOL layers
        model.add(Conv2D(20, 5, padding="same",
                         input_shape=input_shape))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # define the second set of CONV => ACTIVATION => POOL layers
        model.add(Conv2D(50, 5, padding="same"))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # define the first FC => ACTIVATION layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation(activation))

        # define the second FC layer
        model.add(Dense(num_classes))

        # lastly, define the soft-max classifier
        model.add(Activation("softmax"))

        # if a weights path is supplied (indicating that the model was
        # pre-trained), then load the weights
        if weights_path is not None:
            model.load_weights(weights_path)

        return model
