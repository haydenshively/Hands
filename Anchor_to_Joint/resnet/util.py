from tensorflow.keras import layers

def conv1x1(filters, stride=(1,1)):
    return layers.Conv2D(filters=filters, kernel_size=(1,1), strides=stride, kernel_initializer='he_normal')
def conv3x3(filters, stride=(1,1), dilation=1):
    return layers.Conv2D(filters=filters, kernel_size=(3,3), strides=stride, dilation_rate=dilation, padding='same', kernel_initializer='he_normal')
