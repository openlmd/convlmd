__author__ = "Adrián Pallas Fernández"
__email__ = "adrian.pallas@aimen.es"


from keras import backend as K
from keras.models import Model
from keras.layers import Dropout,Conv2D,Activation,MaxPooling2D,Input,Flatten,Dense
from keras.engine.topology import get_source_inputs



def cyplam_model(input_tensor=None,input_shape=None,classes=2):

    if K.image_data_format() == 'channels_last':
        bn_axis = 2
    else:
        bn_axis = 1

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    with K.name_scope('Conv1'):
        x = conv_block2(img_input,5,1,32)
    with K.name_scope('Conv2'):
        x = conv_block2(x,7,2,64)
    with K.name_scope('FC1'):
        x = Flatten(name='flatten')(x)
        x = Dense(1024, activation='relu', name='fc1')(x)
        x = Dropout(0.5)(x)
    with K.name_scope('FC2'):
        x = Dense(classes, activation='relu', name='fc2')(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x, name='cyplam')

    return model


def conv_block2(input_tensor,kernel_size,block,filters,stride=1,pool=2):

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(filters, kernel_size, strides=stride, padding='valid', name='conv2_' + str(block))(input_tensor)
    x = MaxPooling2D(pool,padding='same')(x)
    x = Activation('relu')(x)

    return x

